# WERSJA: 0.5.25
# MAGDA SIP AGENT
#
# Demonstracyjny agent głosowy SIP z Silero VAD i faster-whisper.
# Dane SIP są pobierane ze zmiennych środowiskowych.
# Dane w demo_pesel.txt są wyłącznie przykładowe i pozostają lokalne.

import os
import re
import time
import threading
import unicodedata
import uuid
import queue
import signal
from datetime import datetime
from pathlib import Path
import fcntl
import numpy as np
import onnxruntime as ort

import pjsua2 as pj

from faster_whisper import WhisperModel
from rapidfuzz import fuzz


ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"


def red_text(text):

    if os.environ.get("NO_COLOR"):
        return text

    return f"{ANSI_RED}{text}{ANSI_RESET}"


def caller_id_from_uri(remote_uri):

    value = str(remote_uri or "").strip()

    match = re.search(
        r"(?:sip:|tel:)([^@;>]+)",
        value,
        flags=re.IGNORECASE
    )

    caller_id = (
        match.group(1)
        if match
        else "unknown"
    )

    caller_id = re.sub(
        r"[^0-9A-Za-z+_-]+",
        "_",
        caller_id
    ).strip("_")

    return caller_id or "unknown"


# ============================================================
# KONFIGURACJA
# ============================================================

SIP_SERVER = os.environ["SIP_SERVER"]
SIP_USER = os.environ["SIP_USER"]
SIP_PASSWORD = os.environ["SIP_PASSWORD"]

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
MODELS_DIR = BASE_DIR / "models"
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

WELCOME_FILE = str(AUDIO_DIR / "welcome.wav")
HELP_FILE = str(AUDIO_DIR / "help.wav")
REPEAT_FILE = str(AUDIO_DIR / "repeat.wav")

PESEL_FILE = str(AUDIO_DIR / "pesel.wav")
PESEL_ERROR_FILE = str(AUDIO_DIR / "pesel_error.wav")
PESEL_HELP_FILE = str(AUDIO_DIR / "pesel_help.wav")
INSTALL_CODE_FILE = str(AUDIO_DIR / "install_code.wav")
INSTALL_CODE_ERROR_FILE = str(AUDIO_DIR / "install_code_error.wav")
INSTALL_CODE_HELP_FILE = str(AUDIO_DIR / "install_code_help.wav")
FAULT_TYPE_FILE = str(AUDIO_DIR / "fault_type.wav")

ANALYSIS_START_FILE = str(AUDIO_DIR / "analysis_start.wav")
ANALYSIS_MUSIC_FILE = str(AUDIO_DIR / "analysis_music.wav")
ANALYSIS_DONE_FILE = str(AUDIO_DIR / "analysis_done.wav")

VAD_MODEL = str(MODELS_DIR / "silero_vad.onnx")
VAD_SAMPLE_RATE = 8000
VAD_FRAME_SAMPLES = 256
VAD_THRESHOLD = 0.50
VAD_MIN_SPEECH_MS = 250
VAD_END_SILENCE_MS = 600
VAD_START_TIMEOUT_SEC = 8
MAX_LISTEN_SECONDS = 15
DTMF_INPUT_TIMEOUT_SEC = 15
MAX_PROBLEM_ATTEMPTS = 3

DEMO_VERIFICATION_FILE = BASE_DIR / "demo_pesel.txt"


def load_demo_verification():

    values = {}

    with DEMO_VERIFICATION_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            key, separator, value = line.partition("=")

            if separator:
                values[key.strip()] = value.strip()

    pesel_last4 = values.get("PESEL_LAST4", "")
    install_code = values.get("INSTALL_CODE", "")

    if not re.fullmatch(r"\d{4}", pesel_last4):
        raise RuntimeError(
            "demo_pesel.txt: PESEL_LAST4 musi mieć 4 cyfry"
        )

    if not re.fullmatch(r"\d{5}", install_code):
        raise RuntimeError(
            "demo_pesel.txt: INSTALL_CODE musi mieć 5 cyfr"
        )

    return pesel_last4, install_code


DEMO_PESEL_LAST4, DEMO_INSTALL_CODE = load_demo_verification()

FUZZY_WORD_THRESHOLD = 62
FUZZY_PHRASE_THRESHOLD = 65

HANGUP_DELAY = 0.7
WELCOME_DELAY_SEC = 0.4


# ============================================================
# GLOBALNA KOLEJKA WYNIKÓW STT
# ============================================================

stt_results = queue.Queue()


class SileroVad:
    """Stan modelu jest osobny dla każdego nagrania."""

    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            VAD_MODEL, sess_options=options,
            providers=["CPUExecutionProvider"]
        )
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 32), dtype=np.float32)
        self.pending = np.empty(0, dtype=np.int16)
        self.speech_ms = 0
        self.silence_ms = 0
        self.speech_started = False
        self.ended = False

    def feed(self, pcm):
        samples = np.frombuffer(pcm, dtype="<i2")
        self.pending = np.concatenate((self.pending, samples))
        while len(self.pending) >= VAD_FRAME_SAMPLES and not self.ended:
            chunk = self.pending[:VAD_FRAME_SAMPLES]
            self.pending = self.pending[VAD_FRAME_SAMPLES:]
            audio = (chunk.astype(np.float32) / 32768.0).reshape(1, -1)
            model_input = np.concatenate((self.context, audio), axis=1)
            result, self.state = self.session.run(
                ["output", "stateN"],
                {"input": model_input, "state": self.state,
                 "sr": np.array(VAD_SAMPLE_RATE, dtype=np.int64)}
            )
            self.context = model_input[:, -32:]
            frame_ms = VAD_FRAME_SAMPLES * 1000 // VAD_SAMPLE_RATE
            if float(result[0][0]) >= VAD_THRESHOLD:
                self.speech_ms += frame_ms
                self.silence_ms = 0
                if self.speech_ms >= VAD_MIN_SPEECH_MS:
                    self.speech_started = True
            elif self.speech_started:
                self.silence_ms += frame_ms
                if self.silence_ms >= VAD_END_SILENCE_MS:
                    self.ended = True
            else:
                self.speech_ms = 0


class VadAudioPort(pj.AudioMediaPort):
    """Callback audio tylko kopiuje PCM; wnioskowanie wykonuje tick()."""

    def __init__(self, frames):
        super().__init__()
        self.frames = frames
        self.error = None

    def onFrameReceived(self, frame):
        try:
            data = bytes(frame.buf)
            if data:
                self.frames.put_nowait(data)
        except queue.Full:
            pass
        except Exception as e:
            self.error = str(e)


# ============================================================
# INTENCJE
# ============================================================

AWARIA_WORDS = [
    "awaria",
    "awarie",
    "awarii",
    "awarię",
    "usterka",
    "usterkę",
    "problem",
    "zepsute",
    "zepsuło",
    "zepsul",
]

AWARIA_PHRASES = [
    "chcę zgłosić awarię",
    "chciałbym zgłosić awarię",
    "chciałabym zgłosić awarię",
    "zgłosić awarię",
    "mam awarię",
    "mam problem",
    "mam usterkę",
    "chcę zgłosić usterkę",
    "nie działa internet",
    "internet nie działa",
    "nie mam internetu",
    "brak internetu",
    "nie działa telewizja",
    "telewizja nie działa",
    "brak telewizji",
]

REPORT_WORDS = [
    "zgłosić",
    "zgłaszam",
    "zgłoszenie",
    "zgłosił",
    "zgłosiła",
]


# ============================================================
# START
# ============================================================

def _agent_processes():
    """Znajdź procesy Pythona uruchomione z dokładnie tego pliku."""
    script = Path(__file__).resolve()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            exe = Path(os.readlink(proc / "exe")).name.lower()
            if "python" not in exe:
                continue
            args = (proc / "cmdline").read_bytes().split(b"\0")
            cwd = Path(os.readlink(proc / "cwd"))
            for raw in args[1:]:
                arg = os.fsdecode(raw)
                if not arg.endswith(".py"):
                    continue
                candidate = Path(arg)
                if not candidate.is_absolute():
                    candidate = cwd / candidate
                if candidate.resolve() == script:
                    yield int(proc.name), proc.stat().st_uid
                    break
        except (OSError, ValueError):
            continue  # Proces mógł się zakończyć w trakcie odczytu.


def stop_previous_agent():
    """Ostatni start zastępuje starszą instancję tego samego użytkownika."""
    lock_path = (Path(__file__).resolve().parent /
                 f".agent-start-{os.getuid()}.lock")
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            for pid, uid in list(_agent_processes()):
                if uid != os.getuid():
                    raise RuntimeError(
                        f"agent.py działa jako inny użytkownik (PID {pid}); "
                        "zakończ ten proces z jego konta"
                    )
                print(f"[START] Zatrzymuję poprzedni agent.py (PID {pid})")
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    if not any(old_pid == pid for old_pid, _ in _agent_processes()):
                        break
                    time.sleep(0.1)
                else:
                    raise RuntimeError(
                        f"Poprzedni agent.py (PID {pid}) nie zakończył się "
                        "po SIGTERM; nowa instancja nie wystartuje"
                    )
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


stop_previous_agent()

print()
print("==============================================")
print(" MAGDA SIP AGENT")
print(" WERSJA 0.5.25")
print("==============================================")
print()


# ============================================================
# AUDIO FILE CHECK
# ============================================================

REQUIRED_FILES = [
    WELCOME_FILE,
    HELP_FILE,
    REPEAT_FILE,
    PESEL_FILE,
    PESEL_ERROR_FILE,
    PESEL_HELP_FILE,
    INSTALL_CODE_FILE,
    INSTALL_CODE_ERROR_FILE,
    INSTALL_CODE_HELP_FILE,
    FAULT_TYPE_FILE,
    ANALYSIS_START_FILE,
    ANALYSIS_MUSIC_FILE,
    ANALYSIS_DONE_FILE,
]

for filename in REQUIRED_FILES:

    if not os.path.exists(filename):

        print(
            f"[ERROR] Brak pliku: {filename}"
        )

        raise SystemExit(1)


# ============================================================
# WHISPER
# ============================================================

print("[WHISPER] Ładowanie modelu base...")

whisper = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    local_files_only=True,
)

print("[WHISPER] Model base gotowy")

# Pierwsza transkrypcja jest często znacznie wolniejsza od następnych.
# Wykonujemy ją przed uruchomieniem SIP, aby rozmówca nie czekał na
# inicjalizację dekodera po wypowiedzi. Wynik jest odrzucany.
try:
    warmup_start = time.monotonic()
    warmup_segments, _ = whisper.transcribe(
        WELCOME_FILE,
        language="pl",
        beam_size=1,
        vad_filter=False,
        temperature=0.0,
    )
    for _ in warmup_segments:
        pass
    print(f"[WHISPER] Rozgrzany w {time.monotonic() - warmup_start:.2f} s")
except Exception as e:
    print(f"[WHISPER] Rozgrzewka pominięta: {e}")


# ============================================================
# WORKER WHISPER
#
# WAŻNE:
# Ta funkcja NIE przyjmuje self.
# Nie zna MyCall.
# Nie dotyka PJSUA2.
# Nie wykonuje żadnych metod pj.Call.
# ============================================================

def stt_worker(
    session_id,
    stt_type,
    record_file
):

    start = time.time()

    print(
        f"[STT-WORKER {session_id}] "
        f"Start STT ({stt_type})"
    )

    try:

        if stt_type == "problem":

            prompt = (
                "Klient infolinii może "
                "zgłaszać awarię, usterkę, "
                "problem z internetem "
                "lub telewizją."
            )

            beam_size = 1

        elif stt_type in (
            "pesel",
            "install_code"
        ):

            prompt = None

            beam_size = 5

        elif stt_type == "fault_type":

            prompt = None

            beam_size = 1

        else:

            prompt = None
            beam_size = 1

        print(
            f"[STT-WORKER {session_id}] "
            f"beam_size={beam_size}"
        )

        decoding_options = {}

        if stt_type == "problem":
            decoding_options = {
                "temperature": 0.0,
                "best_of": 1,
            }

        segments, info = whisper.transcribe(
            record_file,
            language="pl",
            beam_size=beam_size,
            vad_filter=True,
            initial_prompt=prompt,
            **decoding_options,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        elapsed = (
            time.time()
            -
            start
        )

        print()
        print(
            "=============================================="
        )

        if stt_type in ("pesel", "install_code"):
            visible_text = red_text(
                "[DANE WERYFIKACYJNE UKRYTE]"
            )
        else:
            visible_text = red_text(text)

        print(
            f"[STT-WORKER {session_id}] "
            f"[STT] {visible_text}"
        )

        print(
            f"[STT-WORKER {session_id}] "
            f"[WHISPER] czas: "
            f"{elapsed:.2f} s"
        )

        print(
            "=============================================="
        )
        print()

        stt_results.put(
            (
                session_id,
                stt_type,
                text,
                None
            )
        )

    except Exception as e:

        print(
            f"[STT-WORKER {session_id}] "
            f"[ERROR] {e}"
        )

        stt_results.put(
            (
                session_id,
                stt_type,
                "",
                str(e)
            )
        )


# ============================================================
# NORMALIZACJA TEKSTU
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = re.sub(
        r"[^\wąćęłńóśźż]+",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# FUZZY
# ============================================================

def best_word_match(
    words,
    patterns
):

    best_score = 0
    best_word = ""
    best_pattern = ""

    for word in words:

        for pattern in patterns:

            score = fuzz.ratio(
                word,
                pattern
            )

            if score > best_score:

                best_score = score
                best_word = word
                best_pattern = pattern

    return (
        best_score,
        best_word,
        best_pattern
    )


def best_phrase_match(
    text,
    patterns
):

    best_score = 0
    best_pattern = ""

    for pattern in patterns:

        score = fuzz.WRatio(
            text,
            pattern
        )

        if score > best_score:

            best_score = score
            best_pattern = pattern

    return (
        best_score,
        best_pattern
    )


# ============================================================
# ROUTER
# ============================================================

def detect_intent(text):

    normalized = normalize_text(
        text
    )

    print(
        f"[ROUTER] Tekst: {normalized}"
    )

    if not normalized:

        return (
            "UNKNOWN",
            0
        )

    words = normalized.split()

    (
        awaria_score,
        awaria_word,
        awaria_pattern
    ) = best_word_match(
        words,
        AWARIA_WORDS
    )

    (
        report_score,
        report_word,
        report_pattern
    ) = best_word_match(
        words,
        REPORT_WORDS
    )

    (
        phrase_score,
        phrase_pattern
    ) = best_phrase_match(
        normalized,
        AWARIA_PHRASES
    )

    print(
        f"[FUZZY] AWARIA word: "
        f"{awaria_score:.1f}% "
        f"'{awaria_word}' ~ "
        f"'{awaria_pattern}'"
    )

    print(
        f"[FUZZY] REPORT word: "
        f"{report_score:.1f}% "
        f"'{report_word}' ~ "
        f"'{report_pattern}'"
    )

    print(
        f"[FUZZY] PHRASE: "
        f"{phrase_score:.1f}% "
        f"~ '{phrase_pattern}'"
    )

    if (
        awaria_score
        >= FUZZY_WORD_THRESHOLD
    ):

        print(
            "[ROUTER] AWARIA przez fuzzy-word"
        )

        return (
            "AWARIA",
            awaria_score
        )

    if (
        report_score >= 75
        and
        phrase_score
        >= FUZZY_PHRASE_THRESHOLD
    ):

        print(
            "[ROUTER] AWARIA przez "
            "REPORT + PHRASE"
        )

        return (
            "AWARIA",
            phrase_score
        )

    if phrase_score >= 82:

        print(
            "[ROUTER] AWARIA przez phrase"
        )

        return (
            "AWARIA",
            phrase_score
        )

    return (
        "UNKNOWN",
        max(
            awaria_score,
            phrase_score
        )
    )


# ============================================================
# PLAYER
# ============================================================

class OneShotPlayer(
    pj.AudioMediaPlayer
):

    def __init__(
        self,
        call,
        purpose,
        session_id
    ):

        super().__init__()

        self.call = call
        self.purpose = purpose
        self.session_id = session_id
        self.eof_sent = False


    def onEof2(self):

        if self.eof_sent:
            return

        self.eof_sent = True

        if not self.call.call_active:

            print(
                f"[AUDIO] Ignoruję EOF "
                f"starej rozmowy: "
                f"{self.purpose}"
            )

            return

        if (
            self.session_id
            != self.call.session_id
        ):

            print(
                "[AUDIO] Ignoruję EOF "
                "nieaktualnej sesji"
            )

            return

        print(
            f"[AUDIO] Koniec pliku: "
            f"{self.purpose}"
        )

        self.call.audio_finished = (
            self.purpose,
            self.session_id
        )


# ============================================================
# CALL
# ============================================================

class MyCall(pj.Call):

    def __init__(
        self,
        account,
        call_id=pj.PJSUA_INVALID_ID
    ):

        super().__init__(
            account,
            call_id
        )

        self.account_ref = account

        self.session_id = (
            uuid.uuid4().hex[:8]
        )

        self.call_active = True
        self.busy_rejected = False
        self.cleanup_done = False

        print(
            f"[SESSION {self.session_id}] "
            f"Nowa rozmowa"
        )

        self.audio_media = None
        self.player = None
        self.recorder = None
        self.vad_port = None
        self.vad_frames = None
        self.vad = None
        self.vad_failed = False

        self.media_initialized = False
        self.confirmed_at = None
        self.welcome_started = False

        self.dialog_state = "WELCOME"

        self.audio_finished = None

        self.listening = False
        self.listen_started = None
        self.current_stt_type = None

        self.processing = False

        self.problem_attempts = 0

        self.hangup_at = None

        self.call_started_at = datetime.now()
        self.caller_id = "unknown"
        self.remote_uri = ""
        self.recording_index = 0
        self.call_recordings_dir = None
        self.record_file = None

        self.dtmf_events = queue.Queue()
        self.dtmf_mode = None
        self.dtmf_buffer = ""
        self.dtmf_started_at = None


    def onDtmfDigit(
        self,
        prm
    ):

        try:
            self.dtmf_events.put_nowait(
                str(prm.digit)
            )
        except Exception as e:
            self.log(
                f"[DTMF CALLBACK ERROR] {e}"
            )


    def set_remote_uri(
        self,
        remote_uri
    ):

        self.remote_uri = str(remote_uri or "")
        self.caller_id = caller_id_from_uri(
            self.remote_uri
        )


    def prepare_record_file(
        self,
        stt_type
    ):

        if self.call_recordings_dir is None:
            day_dir = (
                RECORDINGS_DIR
                /
                self.call_started_at.strftime("%Y-%m-%d")
            )

            call_dir_name = (
                f"{self.call_started_at.strftime('%H-%M-%S')}_"
                f"{self.caller_id}_"
                f"{self.session_id}"
            )

            self.call_recordings_dir = (
                day_dir
                /
                call_dir_name
            )

            self.call_recordings_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            self.log(
                f"[REC] Katalog rozmowy: "
                f"{self.call_recordings_dir}"
            )

        self.recording_index += 1

        safe_stt_type = re.sub(
            r"[^0-9A-Za-z_-]+",
            "_",
            str(stt_type)
        ).strip("_") or "audio"

        self.record_file = str(
            self.call_recordings_dir
            /
            f"{self.recording_index:02d}_{safe_stt_type}.wav"
        )


    # ========================================================
    # LOG
    # ========================================================

    def log(
        self,
        text
    ):

        print(
            f"[SESSION {self.session_id}] "
            f"{text}"
        )


    # ========================================================
    # CALL STATE
    # ========================================================

    def onCallState(
        self,
        prm
    ):

        try:

            info = self.getInfo()

            self.log(
                f"[CALL] "
                f"state={info.stateText} "
                f"code={info.lastStatusCode} "
                f"reason={info.lastReason}"
            )

            if (
                info.state == pj.PJSIP_INV_STATE_CONFIRMED
                and self.confirmed_at is None
            ):
                self.confirmed_at = time.monotonic()

            if (
                info.state
                == pj.PJSIP_INV_STATE_DISCONNECTED
            ):

                self.log(
                    "[CALL] Rozmowa zakończona"
                )

                self.call_active = False

                self.dialog_state = (
                    "DISCONNECTED"
                )

                self.audio_finished = None

                # Połączenia PJSIP odpinane są w cleanup() w pętli głównej.
                self.listening = False
                self.listen_started = None

                self.hangup_at = None

        except Exception as e:

            print(
                "[CALL STATE ERROR]",
                e
            )


    # ========================================================
    # MEDIA
    # ========================================================

    def onCallMediaState(
        self,
        prm
    ):

        if not self.call_active:
            return

        try:

            info = self.getInfo()

            for i, media_info in enumerate(
                info.media
            ):

                if (
                    media_info.type
                    == pj.PJMEDIA_TYPE_AUDIO
                    and
                    media_info.status
                    == pj.PJSUA_CALL_MEDIA_ACTIVE
                ):

                    if self.media_initialized:
                        return

                    self.audio_media = (
                        self.getAudioMedia(i)
                    )
                    self.media_initialized = True

                    self.log(
                        "[AUDIO] Media aktywne"
                    )

                    self.log("[AUDIO] Czekam na potwierdzenie rozmowy przed powitaniem")

        except Exception as e:

            self.log(
                f"[MEDIA ERROR] {e}"
            )


    # ========================================================
    # PLAY
    # ========================================================

    def play_audio(
        self,
        filename,
        purpose
    ):

        if not self.call_active:

            self.log(
                f"[AUDIO] IGNORUJĘ PLAY "
                f"{purpose} - rozmowa zakończona"
            )

            return

        if self.audio_media is None:

            self.log(
                "[AUDIO] Brak aktywnych mediów"
            )

            return

        self.log(
            f"[AUDIO] Odtwarzam: "
            f"{os.path.basename(filename)}"
        )

        if self.player is not None:

            try:

                self.player.stopTransmit(
                    self.audio_media
                )

            except Exception:

                pass

            self.player = None

        self.audio_finished = None

        self.player = OneShotPlayer(
            self,
            purpose,
            self.session_id
        )

        self.player.createPlayer(
            filename,
            pj.PJMEDIA_FILE_NO_LOOP
        )

        self.player.startTransmit(
            self.audio_media
        )


    # ========================================================
    # START LISTENING
    # ========================================================

    def start_listening(
        self,
        stt_type
    ):

        if not self.call_active:
            return

        if self.listening:

            self.log(
                "[REC] Recorder już działa"
            )

            return

        if self.processing:

            self.log(
                "[REC] Whisper jeszcze pracuje"
            )

            return

        print()
        print(
            "=============================================="
        )

        if stt_type == "problem":

            print(
                f"[SESSION {self.session_id}] "
                "[SŁUCHAM] W czym mogę pomóc?"
            )

        elif stt_type == "pesel":

            print(
                f"[SESSION {self.session_id}] "
                "[SŁUCHAM] Podaj 4 ostatnie cyfry PESEL"
            )

        elif stt_type == "install_code":

            print(
                f"[SESSION {self.session_id}] "
                "[SŁUCHAM] Podaj kod miejscowości"
            )

        elif stt_type == "fault_type":

            print(
                f"[SESSION {self.session_id}] "
                "[SŁUCHAM] Internet czy telewizja?"
            )

        print(
            "=============================================="
        )
        print()

        self.prepare_record_file(
            stt_type
        )

        try:

            if os.path.exists(
                self.record_file
            ):

                os.remove(
                    self.record_file
                )

        except Exception as e:

            self.log(
                f"[REC ERROR] {e}"
            )

        self.recorder = (
            pj.AudioMediaRecorder()
        )

        self.recorder.createRecorder(
            self.record_file
        )

        self.audio_media.startTransmit(
            self.recorder
        )

        try:
            self.vad = SileroVad()
            self.vad_frames = queue.Queue(maxsize=100)
            self.vad_port = VadAudioPort(self.vad_frames)
            fmt = pj.MediaFormatAudio()
            fmt.init(pj.PJMEDIA_FORMAT_PCM, VAD_SAMPLE_RATE, 1, 20000, 16)
            self.vad_port.createPort(f"vad_{self.session_id}", fmt)
            self.audio_media.startTransmit(self.vad_port)
            self.vad_failed = False
        except Exception as e:
            self.log(f"[VAD ERROR] {e}; aktywny limit czasu")
            if self.vad_port is not None:
                try:
                    self.audio_media.stopTransmit(self.vad_port)
                except Exception:
                    pass
            self.vad_port = None
            self.vad_frames = None
            self.vad = None
            self.vad_failed = True

        self.listening = True
        self.listen_started = time.monotonic()

        if stt_type == "problem":

            self.dialog_state = (
                "LISTEN_PROBLEM"
            )

        elif stt_type == "pesel":

            self.dialog_state = (
                "LISTEN_PESEL"
            )

        elif stt_type == "install_code":

            self.dialog_state = (
                "LISTEN_INSTALL_CODE"
            )

        elif stt_type == "fault_type":

            self.dialog_state = (
                "LISTEN_FAULT_TYPE"
            )

        # Typ STT zapisujemy jako zwykły string.
        self.current_stt_type = stt_type

        self.log(
            f"[REC] Nagrywam maksymalnie "
            f"{MAX_LISTEN_SECONDS} s"
        )


    # ========================================================
    # DTMF - KLAWIATURA TELEFONU
    # ========================================================

    def stop_voice_for_dtmf(
        self
    ):

        if not self.listening:
            return

        self.listening = False
        self.listen_started = None

        if self.vad_port is not None and self.audio_media is not None:
            try:
                self.audio_media.stopTransmit(
                    self.vad_port
                )
            except Exception:
                pass

        self.vad_port = None
        self.vad_frames = None
        self.vad = None

        if self.recorder is not None and self.audio_media is not None:
            try:
                self.audio_media.stopTransmit(
                    self.recorder
                )
            except Exception:
                pass

        self.recorder = None

        self.log(
            "[DTMF] Wykryto klawiaturę telefonu; "
            "rozpoznawanie głosu zatrzymane"
        )


    def reset_dtmf(
        self
    ):

        self.dtmf_mode = None
        self.dtmf_buffer = ""
        self.dtmf_started_at = None


    def finish_dtmf(
        self
    ):

        mode = self.dtmf_mode
        digits = self.dtmf_buffer
        expected_length = (
            4
            if mode == "pesel"
            else 5
        )

        self.reset_dtmf()

        if len(digits) != expected_length:
            self.log(
                f"[DTMF] Nieprawidłowa liczba cyfr: "
                f"{len(digits)}/{expected_length}"
            )

            if mode == "pesel":
                self.dialog_state = "PESEL_ERROR"
                self.play_audio(
                    PESEL_ERROR_FILE,
                    "pesel_error"
                )
            else:
                self.dialog_state = "INSTALL_CODE_ERROR"
                self.play_audio(
                    INSTALL_CODE_ERROR_FILE,
                    "install_code_error"
                )

            return

        self.log(
            f"[DTMF] Zatwierdzono {expected_length} "
            f"{'cyfry' if expected_length == 4 else 'cyfr'}"
        )

        if mode == "pesel":
            self.verify_pesel_digits(
                digits
            )
        else:
            self.verify_install_code_digits(
                digits
            )


    def process_dtmf_events(
        self
    ):

        while True:
            try:
                digit = self.dtmf_events.get_nowait()
            except queue.Empty:
                break

            if self.dtmf_mode is None:
                if (
                    not self.listening
                    or self.current_stt_type
                    not in ("pesel", "install_code")
                ):
                    self.log(
                        "[DTMF] Znak zignorowany poza "
                        "weryfikacją"
                    )
                    continue

                self.dtmf_mode = self.current_stt_type
                self.dtmf_buffer = ""
                self.dtmf_started_at = time.monotonic()
                self.stop_voice_for_dtmf()

                if self.dtmf_mode == "pesel":
                    self.dialog_state = "WAIT_DTMF_PESEL"
                else:
                    self.dialog_state = "WAIT_DTMF_INSTALL_CODE"

            self.dtmf_started_at = time.monotonic()

            if digit.isdigit():
                expected_length = (
                    4
                    if self.dtmf_mode == "pesel"
                    else 5
                )

                if len(self.dtmf_buffer) < expected_length:
                    self.dtmf_buffer += digit
                    self.log(
                        f"[DTMF] Odebrano cyfrę "
                        f"{len(self.dtmf_buffer)}/{expected_length}"
                    )
                else:
                    self.log(
                        "[DTMF] Nadmiarowa cyfra zignorowana"
                    )

                continue

            if digit == "*":
                self.dtmf_buffer = ""
                self.log(
                    "[DTMF] Wpis wyczyszczony"
                )
                continue

            if digit == "#":
                self.finish_dtmf()
                return

            self.log(
                "[DTMF] Nieobsługiwany znak zignorowany"
            )


    # ========================================================
    # STOP LISTENING
    # ========================================================

    def stop_listening(
        self
    ):

        if not self.listening:
            return

        self.listening = False

        if self.vad_port is not None and self.audio_media is not None:
            try:
                self.audio_media.stopTransmit(self.vad_port)
            except Exception as e:
                self.log(f"[VAD stopTransmit] {e}")
        self.vad_port = None
        self.vad_frames = None
        self.vad = None

        self.log(
            "[REC] Koniec nagrywania"
        )

        stt_type = self.current_stt_type

        try:

            if (
                self.recorder is not None
                and
                self.audio_media is not None
            ):

                self.audio_media.stopTransmit(
                    self.recorder
                )

        except Exception as e:

            self.log(
                f"[REC stopTransmit] {e}"
            )

        # Bardzo ważne:
        # obiekt PJSUA2 niszczymy tutaj,
        # w głównym wątku.
        self.recorder = None

        self.listen_started = None

        if not self.call_active:

            self.log(
                "[REC] Rozmowa zakończona "
                "- nie uruchamiam STT"
            )

            return

        time.sleep(
            0.15
        )

        try:

            size = os.path.getsize(
                self.record_file
            )

            self.log(
                f"[REC] WAV size: "
                f"{size} bytes"
            )

        except Exception as e:

            self.log(
                f"[REC WAV ERROR] {e}"
            )

            return

        if stt_type == "problem":

            self.dialog_state = (
                "WAIT_STT_PROBLEM"
            )

        elif stt_type == "pesel":

            self.dialog_state = (
                "WAIT_STT_PESEL"
            )

        elif stt_type == "install_code":

            self.dialog_state = (
                "WAIT_STT_INSTALL_CODE"
            )

        elif stt_type == "fault_type":

            self.dialog_state = (
                "WAIT_STT_FAULT_TYPE"
            )

        self.processing = True

        # Worker otrzymuje zwykłe dane i nie korzysta z obiektów PJSUA2.

        session_id = str(
            self.session_id
        )

        record_file = str(
            self.record_file
        )

        worker_type = str(
            stt_type
        )

        thread = threading.Thread(
            target=stt_worker,
            args=(
                session_id,
                worker_type,
                record_file
            ),
            daemon=True
        )

        thread.start()


    # ========================================================
    # WYNIK STT
    # ========================================================

    def handle_stt_result(
        self,
        stt_type,
        text,
        error
    ):

        if not self.call_active:
            return

        self.processing = False

        if error:

            self.log(
                f"[WHISPER ERROR] {error}"
            )

        if stt_type in ("pesel", "install_code"):
            self.log(
                "[STT] Wynik odebrany: "
                "[DANE WERYFIKACYJNE UKRYTE]"
            )
        else:
            self.log(
                f"[STT] Wynik odebrany: "
                f"'{red_text(text)}'"
            )

        if stt_type == "problem":

            if (
                self.dialog_state
                != "WAIT_STT_PROBLEM"
            ):

                self.log(
                    "[STT] Ignoruję wynik "
                    "- zły stan dialogu"
                )

                return

            self.handle_problem(
                text
            )

            return

        if stt_type == "pesel":

            if (
                self.dialog_state
                != "WAIT_STT_PESEL"
            ):

                self.log(
                    "[STT] Ignoruję wynik "
                    "- zły stan dialogu"
                )

                return

            self.handle_pesel(
                text
            )

            return

        if stt_type == "install_code":

            if (
                self.dialog_state
                != "WAIT_STT_INSTALL_CODE"
            ):

                self.log(
                    "[STT] Ignoruję wynik "
                    "- zły stan dialogu"
                )

                return

            self.handle_install_code(
                text
            )

            return

        if stt_type == "fault_type":

            if self.dialog_state != "WAIT_STT_FAULT_TYPE":

                self.log("[STT] Ignoruję wynik - zły stan dialogu")
                return

            self.handle_fault_type(text)
            return


    # ========================================================
    # PROBLEM
    # ========================================================

    def handle_problem(
        self,
        text
    ):

        if not self.call_active:
            return

        help_word = normalize_text(text)
        help_score = max(
            fuzz.ratio(help_word, expected)
            for expected in ("pomoc", "pomóc", "pomocy")
        )

        if " " not in help_word and help_score >= 80:

            self.problem_attempts = 0
            self.dialog_state = "HELP"
            self.log(f"[INTENT] POMOC score={help_score:.1f}")
            self.play_audio(HELP_FILE, "help")
            return

        intent, score = detect_intent(
            text
        )

        self.log(
            f"[INTENT] "
            f"{intent} "
            f"score={score:.1f}"
        )

        if intent == "AWARIA":

            print()
            print(
                "[INTENT] >>> AWARIA <<<"
            )
            print()

            self.problem_attempts = 0

            self.dialog_state = (
                "ASK_PESEL"
            )

            self.log(
                "[MAGDA] W celu weryfikacji podaj "
                "cztery ostatnie cyfry numeru PESEL. "
                "Możesz je powiedzieć, cyfra po cyfrze, "
                "lub wpisać na klawiaturze telefonu, "
                "zatwierdzając kratką. Jeśli potrzebujesz "
                "podpowiedzi, powiedz pomoc."
            )

            self.play_audio(
                PESEL_FILE,
                "ask_pesel"
            )

            return

        self.problem_attempts += 1

        self.log(
            f"[ROUTER] Nie rozpoznano. "
            f"Próba "
            f"{self.problem_attempts}/"
            f"{MAX_PROBLEM_ATTEMPTS}"
        )

        if (
            self.problem_attempts
            >= MAX_PROBLEM_ATTEMPTS
        ):

            self.log(
                "[DIALOG] Limit prób"
            )

            self.dialog_state = (
                "FAILED_PROBLEM"
            )

            return

        self.dialog_state = (
            "REPEAT_PROBLEM"
        )

        self.log(
            "[MAGDA] Przepraszam, "
            "nie zrozumiałam. "
            "Czy możesz powtórzyć?"
        )

        self.play_audio(
            REPEAT_FILE,
            "repeat_problem"
        )


    # ========================================================
    # PESEL - PARSER CYFR
    # ========================================================

    def normalize_pesel(
        self,
        text
    ):

        text = text.lower()

        text = unicodedata.normalize(
            "NFKC",
            text
        )

        text = re.sub(
            r"[,.;:!?()\[\]{}]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        self.log(
            "[PARSER CYFR] Analizuję "
            "ukrytą odpowiedź"
        )

        number_words = {

            "zero": "0",

            "jeden": "1",
            "jedna": "1",
            "jedynka": "1",
            "raz": "1",

            # Wariant transkrypcji słowa "raz".
            "roz": "1",

            "dwa": "2",
            "dwójka": "2",
            "dwojka": "2",

            "trzy": "3",
            "trójka": "3",
            "trojka": "3",

            "cztery": "4",
            "czwórka": "4",
            "czworka": "4",

            "pięć": "5",
            "piec": "5",
            "piątka": "5",
            "piatka": "5",

            "sześć": "6",
            "szesc": "6",
            "szóstka": "6",
            "szostka": "6",

            "siedem": "7",
            "siódemka": "7",
            "siodemka": "7",

            "osiem": "8",
            "ósemka": "8",
            "osemka": "8",

            "dziewięć": "9",
            "dziewiec": "9",
            "dziewiątka": "9",
            "dziewiatka": "9",
        }

        result = []

        tokens = text.split()

        for token in tokens:

            # np. 1 / 23 / 12345
            if token.isdigit():

                result.extend(
                    list(token)
                )

                continue

            # np. raz / dwa / trzy
            if token in number_words:

                result.append(
                    number_words[token]
                )

                continue

            # np. 2-3 / 2/3
            token_digits = re.sub(
                r"\D",
                "",
                token
            )

            if token_digits:

                result.extend(
                    list(token_digits)
                )

        pesel = "".join(
            result
        )

        self.log(
            f"[PARSER CYFR] Odczytano "
            f"{len(pesel)} cyfr"
        )

        return pesel


    # ========================================================
    # PESEL
    # ========================================================

    def handle_pesel(
        self,
        text
    ):

        if not self.call_active:
            return

        help_word = normalize_text(text)
        help_score = max(
            fuzz.ratio(help_word, expected)
            for expected in ("pomoc", "pomóc", "pomocy")
        )

        if " " not in help_word and help_score >= 80:
            self.dialog_state = "PESEL_HELP"
            self.log("[WERYFIKACJA] Pomoc przy podawaniu PESEL")
            self.play_audio(PESEL_HELP_FILE, "pesel_help")
            return

        pesel = self.normalize_pesel(
            text
        )

        self.verify_pesel_digits(
            pesel
        )


    def verify_pesel_digits(
        self,
        pesel
    ):

        self.log(
            f"[WERYFIKACJA] PESEL4: "
            f"odczytano {len(pesel)} cyfry"
        )

        if pesel == DEMO_PESEL_LAST4:

            print()
            print(
                "[WERYFIKACJA] >>> OK <<<"
            )
            print()

            self.dialog_state = "ASK_INSTALL_CODE"
            self.log(
                "[MAGDA] PESEL potwierdzony. "
                "Teraz podaj pięciocyfrowy kod miejscowości, "
                "w której znajduje się instalacja. Możesz go "
                "powiedzieć, cyfra po cyfrze, lub wpisać na "
                "klawiaturze telefonu, zatwierdzając kratką. "
                "Jeśli nie wiesz, gdzie go znaleźć, powiedz pomoc."
            )
            self.play_audio(
                INSTALL_CODE_FILE,
                "ask_install_code"
            )

            return

        print()
        print(
            "[WERYFIKACJA] >>> BŁĄD <<<"
        )
        print()

        self.dialog_state = (
            "PESEL_ERROR"
        )

        self.play_audio(
            PESEL_ERROR_FILE,
            "pesel_error"
        )


    def handle_install_code(
        self,
        text
    ):

        if not self.call_active:
            return

        help_word = normalize_text(text)
        help_score = max(
            fuzz.ratio(help_word, expected)
            for expected in ("pomoc", "pomóc", "pomocy")
        )

        if " " not in help_word and help_score >= 80:
            self.dialog_state = "INSTALL_CODE_HELP"
            self.log("[WERYFIKACJA] Pomoc przy kodzie miejscowości")
            self.play_audio(
                INSTALL_CODE_HELP_FILE,
                "install_code_help"
            )
            return

        install_code = self.normalize_pesel(
            text
        )

        self.verify_install_code_digits(
            install_code
        )


    def verify_install_code_digits(
        self,
        install_code
    ):

        self.log(
            f"[WERYFIKACJA] Kod miejscowości: "
            f"odczytano {len(install_code)} cyfr"
        )

        if install_code == DEMO_INSTALL_CODE:

            print()
            print(
                "[WERYFIKACJA] >>> PEŁNA WERYFIKACJA OK <<<"
            )
            print()

            self.dialog_state = "ASK_FAULT_TYPE"
            self.log(
                "[MAGDA] Weryfikacja zakończona. "
                "Czego dotyczy usterka: "
                "internetu czy telewizji?"
            )
            self.play_audio(
                FAULT_TYPE_FILE,
                "ask_fault_type"
            )

            return

        print()
        print(
            "[WERYFIKACJA] >>> BŁĘDNY KOD MIEJSCOWOŚCI <<<"
        )
        print()

        self.dialog_state = "INSTALL_CODE_ERROR"
        self.play_audio(
            INSTALL_CODE_ERROR_FILE,
            "install_code_error"
        )


    def handle_fault_type(self, text):

        if not self.call_active:
            return

        normalized = unicodedata.normalize("NFKD", text.lower())
        normalized = "".join(
            char for char in normalized
            if not unicodedata.combining(char)
        )

        internet = any(word in normalized for word in ("internet", "wi-fi", "wifi"))
        television = any(word in normalized for word in ("telewiz", "tv"))

        if internet and not television:
            choice = "internet"
        elif television and not internet:
            choice = "telewizja"
        elif internet and television:
            choice = None
        else:
            compact = re.sub(r"[^a-z]", "", normalized)
            internet_score = max(
                fuzz.ratio(compact, word)
                for word in ("internet", "iterminat", "nitarna")
            )
            television_score = max(
                fuzz.ratio(compact, word)
                for word in ("telewizja", "telewizji", "telewizor")
            )
            self.log(
                f"[WYBÓR USTERKI] podobieństwo: "
                f"internet={internet_score:.1f}, "
                f"telewizja={television_score:.1f}"
            )
            if max(internet_score, television_score) < 68 or abs(internet_score - television_score) < 20:
                choice = None
            else:
                choice = "internet" if internet_score > television_score else "telewizja"

        if choice is None:
            self.log("[WYBÓR USTERKI] Nie rozpoznano jednoznacznie; pytam ponownie")
            self.dialog_state = "ASK_FAULT_TYPE"
            self.play_audio(FAULT_TYPE_FILE, "ask_fault_type")
            return

        self.fault_type = choice
        self.log(f"[WYBÓR USTERKI] {self.fault_type}")
        self.dialog_state = "ANALYSIS_MUSIC"
        self.play_audio(ANALYSIS_MUSIC_FILE, "analysis_music")


    # ========================================================
    # AUDIO EOF
    # ========================================================

    def handle_audio_finished(
        self
    ):

        if not self.call_active:

            self.audio_finished = None
            return

        event = self.audio_finished

        self.audio_finished = None

        if event is None:
            return

        purpose, session_id = event

        if (
            session_id
            != self.session_id
        ):

            self.log(
                "[EOF] Stara sesja "
                "- IGNORUJĘ"
            )

            return

        self.log(
            f"[EOF] "
            f"purpose={purpose} "
            f"state={self.dialog_state}"
        )

        valid = False

        if (
            purpose == "welcome"
            and
            self.dialog_state == "WELCOME"
        ):

            valid = True

        elif (
            purpose == "repeat_problem"
            and
            self.dialog_state
            == "REPEAT_PROBLEM"
        ):

            valid = True

        elif (
            purpose == "ask_pesel"
            and
            self.dialog_state == "ASK_PESEL"
        ):

            valid = True

        elif (
            purpose == "pesel_error"
            and
            self.dialog_state == "PESEL_ERROR"
        ):

            valid = True

        elif (
            purpose == "pesel_help"
            and
            self.dialog_state == "PESEL_HELP"
        ):

            valid = True

        elif (
            purpose == "ask_install_code"
            and
            self.dialog_state == "ASK_INSTALL_CODE"
        ):

            valid = True

        elif (
            purpose == "install_code_error"
            and
            self.dialog_state == "INSTALL_CODE_ERROR"
        ):

            valid = True

        elif (
            purpose == "install_code_help"
            and
            self.dialog_state == "INSTALL_CODE_HELP"
        ):

            valid = True

        elif (
            purpose == "help"
            and self.dialog_state == "HELP"
        ):

            valid = True

        elif (
            purpose == "ask_fault_type"
            and
            self.dialog_state == "ASK_FAULT_TYPE"
        ):

            valid = True

        elif (
            purpose == "analysis_start"
            and
            self.dialog_state
            == "ANALYSIS_START"
        ):

            valid = True

        elif (
            purpose == "analysis_music"
            and
            self.dialog_state
            == "ANALYSIS_MUSIC"
        ):

            valid = True

        elif (
            purpose == "analysis_done"
            and
            self.dialog_state
            == "ANALYSIS_DONE"
        ):

            valid = True

        if not valid:

            self.log(
                f"[EOF] Ignoruję "
                f"spóźniony callback: "
                f"{purpose}"
            )

            return

        if self.player is not None:

            try:

                self.player.stopTransmit(
                    self.audio_media
                )

            except Exception:

                pass

        # ----------------------------------------------------
        # WELCOME
        # ----------------------------------------------------

        if purpose == "welcome":

            self.log(
                "[DIALOG] "
                "Powitanie zakończone"
            )

            self.dialog_state = (
                "LISTEN_PROBLEM"
            )

            self.start_listening(
                "problem"
            )

            return

        # ----------------------------------------------------
        # REPEAT
        # ----------------------------------------------------

        if purpose == "repeat_problem":

            self.log(
                "[DIALOG] "
                "Ponownie słucham problemu"
            )

            self.dialog_state = (
                "LISTEN_PROBLEM"
            )

            self.start_listening(
                "problem"
            )

            return

        # ----------------------------------------------------
        # ASK PESEL
        # ----------------------------------------------------

        if purpose == "ask_pesel":

            self.log(
                "[DIALOG] Czekam na PESEL"
            )

            self.dialog_state = (
                "LISTEN_PESEL"
            )

            self.start_listening(
                "pesel"
            )

            return

        # ----------------------------------------------------
        # PESEL ERROR
        # ----------------------------------------------------

        if purpose == "pesel_error":

            self.log(
                "[DIALOG] "
                "Ponowna próba PESEL"
            )

            self.dialog_state = (
                "LISTEN_PESEL"
            )

            self.start_listening(
                "pesel"
            )

            return

        if purpose == "pesel_help":

            self.log(
                "[DIALOG] Pomoc PESEL zakończona"
            )
            self.dialog_state = "LISTEN_PESEL"
            self.start_listening("pesel")
            return

        if purpose == "ask_install_code":

            self.log(
                "[DIALOG] Czekam na kod miejscowości"
            )
            self.dialog_state = "LISTEN_INSTALL_CODE"
            self.start_listening("install_code")
            return

        if purpose == "install_code_error":

            self.log(
                "[DIALOG] Ponowna próba kodu miejscowości"
            )
            self.dialog_state = "LISTEN_INSTALL_CODE"
            self.start_listening("install_code")
            return

        if purpose == "install_code_help":

            self.log(
                "[DIALOG] Pomoc kodu miejscowości zakończona"
            )
            self.dialog_state = "LISTEN_INSTALL_CODE"
            self.start_listening("install_code")
            return

        if purpose == "help":

            self.log("[DIALOG] Pomoc zakończona; ponownie słucham sprawy")
            self.dialog_state = "LISTEN_PROBLEM"
            self.start_listening("problem")
            return

        if purpose == "ask_fault_type":

            self.log("[DIALOG] Czekam na wybór rodzaju usterki")
            self.dialog_state = "LISTEN_FAULT_TYPE"
            self.start_listening("fault_type")
            return

        # ----------------------------------------------------
        # ANALIZA START -> MUZYKA
        # ----------------------------------------------------

        if purpose == "analysis_start":

            self.log(
                "[ANALIZA] "
                "Uruchamiam analizę"
            )

            self.dialog_state = (
                "ANALYSIS_MUSIC"
            )

            self.play_audio(
                ANALYSIS_MUSIC_FILE,
                "analysis_music"
            )

            return

        # ----------------------------------------------------
        # MUZYKA -> WYNIK
        # ----------------------------------------------------

        if purpose == "analysis_music":

            self.log(
                "[ANALIZA] "
                "Analiza zakończona"
            )

            self.dialog_state = (
                "ANALYSIS_DONE"
            )

            self.log(
                "[MAGDA] Sprawdziłam twoje "
                "urządzenia i nie widzę "
                "żadnych problemów. "
                "Utworzę zgłoszenie w twojej "
                "sprawie. Oczekuj na kontakt "
                "konsultanta."
            )

            self.play_audio(
                ANALYSIS_DONE_FILE,
                "analysis_done"
            )

            return

        # ----------------------------------------------------
        # KONIEC -> TIMER ROZŁĄCZENIA
        # ----------------------------------------------------

        if purpose == "analysis_done":

            self.log(
                "[DIALOG] "
                "Obsługa zakończona"
            )

            self.dialog_state = (
                "WAIT_HANGUP"
            )

            self.hangup_at = (
                time.time()
                +
                HANGUP_DELAY
            )

            self.log(
                f"[CALL] Automatyczne "
                f"rozłączenie za "
                f"{HANGUP_DELAY:.1f} s"
            )

            return


    # ========================================================
    # HANGUP
    # ========================================================

    def do_hangup(
        self
    ):

        if not self.call_active:
            return

        self.hangup_at = None

        self.log(
            "[CALL] Rozłączam rozmowę"
        )

        try:

            op = pj.CallOpParam()

            op.statusCode = 200

            self.hangup(
                op
            )

        except Exception as e:

            self.log(
                f"[HANGUP ERROR] {e}"
            )


    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(
        self
    ):

        if self.cleanup_done:
            return

        self.cleanup_done = True

        self.call_active = False
        self.listening = False
        self.listen_started = None
        if self.vad_port is not None and self.audio_media is not None:
            try:
                self.audio_media.stopTransmit(self.vad_port)
            except Exception:
                pass
        self.vad_port = None
        self.vad_frames = None
        self.vad = None
        self.audio_finished = None
        self.hangup_at = None
        self.reset_dtmf()

        if (
            self.recorder is not None
            and
            self.audio_media is not None
        ):

            try:

                self.audio_media.stopTransmit(
                    self.recorder
                )

            except Exception:

                pass

        self.recorder = None

        if (
            self.player is not None
            and
            self.audio_media is not None
        ):

            try:

                self.player.stopTransmit(
                    self.audio_media
                )

            except Exception:

                pass

        self.player = None
        self.audio_media = None

        self.log(
            "[CLEANUP] Sesja wyczyszczona"
        )


    # ========================================================
    # TICK
    # ========================================================

    def tick(
        self
    ):

        if not self.call_active:

            self.cleanup()
            return

        if (
            self.dialog_state == "WELCOME"
            and not self.welcome_started
            and self.audio_media is not None
            and self.confirmed_at is not None
            and time.monotonic() - self.confirmed_at >= WELCOME_DELAY_SEC
        ):
            self.play_audio(WELCOME_FILE, "welcome")
            self.welcome_started = True
            return

        # ----------------------------------------------------
        # AUTOMATYCZNE ROZŁĄCZENIE
        # ----------------------------------------------------

        if (
            self.dialog_state == "WAIT_HANGUP"
            and
            self.hangup_at is not None
            and
            time.time() >= self.hangup_at
        ):

            self.do_hangup()

            return

        # ----------------------------------------------------
        # EOF AUDIO
        # ----------------------------------------------------

        if self.audio_finished is not None:

            self.handle_audio_finished()

            return

        # Callback onDtmfDigit() tylko odkłada znaki do kolejki.
        # Odczyt i operacje PJSIP wykonujemy tutaj, w głównym tick().
        self.process_dtmf_events()

        if (
            self.dtmf_mode is not None
            and self.dtmf_started_at is not None
            and time.monotonic() - self.dtmf_started_at
            >= DTMF_INPUT_TIMEOUT_SEC
        ):
            self.log(
                "[DTMF] Minął czas na zatwierdzenie znakiem #"
            )
            self.finish_dtmf()
            return

        # ----------------------------------------------------
        # VAD i timery nagrywania: wyłącznie główny tick().
        # ----------------------------------------------------

        if (
            self.listening
            and
            self.listen_started is not None
        ):

            if self.vad_port is not None and self.vad_port.error:
                self.log(f"[VAD CALLBACK ERROR] {self.vad_port.error}")
                self.vad_port.error = None
                self.vad = None
                self.vad_failed = True

            if self.vad is not None and self.vad_frames is not None:
                try:
                    for _ in range(100):
                        try:
                            frame = self.vad_frames.get_nowait()
                        except queue.Empty:
                            break
                        self.vad.feed(frame)
                        if self.vad.ended:
                            self.log("[VAD] Koniec mowy")
                            self.stop_listening()
                            return
                except Exception as e:
                    self.log(f"[VAD ERROR] {e}; aktywny limit czasu")
                    self.vad = None
                    self.vad_failed = True

            elapsed = time.monotonic() - self.listen_started
            if elapsed >= MAX_LISTEN_SECONDS:
                self.log("[REC] Maksymalny czas nagrania")
                self.stop_listening()
                return
            if (elapsed >= VAD_START_TIMEOUT_SEC
                    and self.vad is not None
                    and not self.vad.speech_started):
                self.log("[VAD] Brak początku mowy")
                self.stop_listening()
                return


# ============================================================
# ACCOUNT
# ============================================================

class MyAccount(pj.Account):

    def __init__(
        self
    ):

        super().__init__()

        self.calls = []
        self.incoming_lock = threading.Lock()
        self.pending_answers = []


    def process_pending_answers(
        self
    ):

        with self.incoming_lock:
            pending = self.pending_answers
            self.pending_answers = []

        for call, status_code in pending:

            try:
                op = pj.CallOpParam()
                op.statusCode = status_code
                call.answer(op)

                if status_code == 486:
                    print(
                        "[SIP] Zajęte: dodatkowe "
                        "połączenie odrzucone (486)"
                    )
                else:
                    print("[SIP] Odebrano połączenie")

            except Exception as e:
                call.log(
                    f"[SIP ERROR] Nie udało się wysłać "
                    f"odpowiedzi {status_code}: {e}"
                )
                call.call_active = False
                call.dialog_state = "DISCONNECTED"


    def onRegState(
        self,
        prm
    ):

        info = self.getInfo()

        print(
            f"[REGISTER] "
            f"active={info.regIsActive} "
            f"status={info.regStatus} "
            f"reason={info.regStatusText}"
        )


    def onIncomingCall(
        self,
        prm
    ):

        print()
        print(
            "=============================================="
        )
        print(
            "[SIP] NOWE POŁĄCZENIE"
        )
        print(
            "=============================================="
        )

        with self.incoming_lock:
            busy = any(
                existing.call_active and not existing.busy_rejected
                and existing.dialog_state != "DISCONNECTED"
                for existing in self.calls
            )
            call = MyCall(self, prm.callId)
            call.busy_rejected = busy
            self.calls.append(call)
            self.pending_answers.append(
                (
                    call,
                    486 if busy else 200
                )
            )

        try:
            info = call.getInfo()
            call.set_remote_uri(info.remoteUri)
            call.log(
                f"[SIP] slot={info.id} Call-ID={info.callIdString} "
                f"from={info.remoteUri} decyzja={'486' if busy else '200'}"
            )
        except Exception as e:
            call.log(f"[SIP] callId={prm.callId} decyzja="
                     f"{'486' if busy else '200'}; getInfo: {e}")

# ============================================================
# ODBIERANIE WYNIKÓW STT
#
# Ta funkcja jest wywoływana z MAIN LOOP.
# Dopiero tutaj wynik workera trafia do MyCall.
# ============================================================

def process_stt_results(
    account
):

    while True:

        try:

            (
                session_id,
                stt_type,
                text,
                error
            ) = stt_results.get_nowait()

        except queue.Empty:

            break

        target_call = None

        for call in account.calls:

            if (
                call.session_id
                == session_id
            ):

                target_call = call
                break

        if target_call is None:

            print(
                f"[STT] Wynik dla starej "
                f"sesji {session_id} "
                f"- IGNORUJĘ"
            )

            continue

        if not target_call.call_active:

            print(
                f"[STT] Rozmowa "
                f"{session_id} już zakończona "
                f"- IGNORUJĘ"
            )

            continue

        target_call.handle_stt_result(
            stt_type,
            text,
            error
        )


# ============================================================
# MAIN
# ============================================================

def main():

    ep = pj.Endpoint()

    ep.libCreate()

    ep_cfg = pj.EpConfig()

    ep_cfg.uaConfig.maxCalls = 8
    ep_cfg.uaConfig.threadCnt = 0
    ep_cfg.uaConfig.mainThreadOnly = True

    ep_cfg.logConfig.level = 3
    ep_cfg.logConfig.consoleLevel = 3

    ep.libInit(
        ep_cfg
    )

    transport_cfg = (
        pj.TransportConfig()
    )

    transport_cfg.port = 5060

    ep.transportCreate(
        pj.PJSIP_TRANSPORT_UDP,
        transport_cfg
    )

    ep.libStart()

    ep.audDevManager().setNullDev()

    print(
        "[PJSIP] Uruchomiony"
    )

    acc_cfg = pj.AccountConfig()

    acc_cfg.idUri = (
        f"sip:{SIP_USER}@{SIP_SERVER}"
    )

    acc_cfg.regConfig.registrarUri = (
        f"sip:{SIP_SERVER}"
    )

    acc_cfg.sipConfig.authCreds.append(
        pj.AuthCredInfo(
            "digest",
            "*",
            SIP_USER,
            0,
            SIP_PASSWORD
        )
    )

    account = MyAccount()

    account.create(
        acc_cfg
    )

    print(
        f"[SIP] Konto: {SIP_USER}"
    )

    print(
        f"[SIP] Proxy: {SIP_SERVER}"
    )

    print(
        "[SIP] Czekam na połączenie..."
    )

    try:

        while True:

            # -----------------------------------------------
            # Wyniki Whispera odbieramy TUTAJ.
            # -----------------------------------------------

            process_stt_results(
                account
            )

            # -----------------------------------------------
            # Tick aktywnych rozmów
            # -----------------------------------------------

            for call in list(
                account.calls
            ):

                try:

                    call.tick()

                except Exception as e:

                    print(
                        f"[TICK ERROR] {e}"
                    )

            # -----------------------------------------------
            # Najpierw cleanup zakończonych rozmów.
            # -----------------------------------------------

            for call in list(
                account.calls
            ):

                if not call.call_active:

                    try:

                        call.cleanup()

                    except Exception as e:

                        print(
                            f"[CLEANUP ERROR] {e}"
                        )

            # -----------------------------------------------
            # Dopiero potem usuwamy je z listy.
            # -----------------------------------------------

            with account.incoming_lock:
                account.calls = [
                    call
                    for call in account.calls
                    if call.call_active
                ]

            # Callbacki SIP działają w tym samym wątku co tick() i cleanup().
            ep.libHandleEvents(20)

            # Odpowiedzi na INVITE są wysyłane po zakończeniu callbacku.
            account.process_pending_answers()

    except KeyboardInterrupt:

        print()
        print(
            "[STOP]"
        )

    finally:

        for call in list(
            account.calls
        ):

            try:

                call.cleanup()

            except Exception:

                pass

        ep.libDestroy()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
