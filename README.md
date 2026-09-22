# Magda SIP Agent

An experimental Python SIP voice agent for learning, testing, and fun.

Magda answers SIP calls, plays IVR prompts, records caller speech, detects the
end of speech with Silero VAD, transcribes Polish audio with faster-whisper,
and follows a small demonstration dialog tree.

## Features

- PJSUA2 SIP registration and incoming calls
- one active call with `486 Busy Here` for additional calls
- WAV playback and per-call recordings
- Silero VAD through ONNX Runtime, without Torch or CUDA
- Polish transcription with faster-whisper on CPU
- DTMF input terminated with `#`
- demonstration verification using four digits and a five-digit location code

## Important limitations

This is a hobby and learning project, not a production contact-center system.
The diagnostic actions and ticket creation are demonstrations. Review your
local telecommunications, privacy, call-recording, and data-protection rules
before using it with real callers.

Do not commit SIP credentials, API keys, recordings, logs, or real customer
data. The supplied verification values are fictional examples.

## Linux quick start

The main agent is designed for `/opt/asystent` and requires a compatible
PJSUA2 Python binding, faster-whisper, NumPy, ONNX Runtime, and RapidFuzz.

Create the environment and install the Python dependencies:

```bash
cd /opt/asystent
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PJSUA2 is platform specific and must be built or installed separately for the
Python version used by the host. Confirm it is available before starting:

```bash
python -c "import pjsua2; print('PJSUA2: OK')"
```

Copy `sip.env.example` to an ignored local file:

```bash
mkdir -p /opt/asystent/secrets
cp sip.env.example /opt/asystent/secrets/sip.env
chmod 600 /opt/asystent/secrets/sip.env
```

Edit the file, then run:

```bash
./scripts/run.sh
```

The repository includes empty `logs/` and `recordings/` directories. Runtime
logs and call recordings stored in them are ignored by Git.

## Audio and models

Sample Polish WAV prompts and the Silero VAD ONNX model are included. Whisper
models are downloaded on first use and are not stored in this repository.

### Creating a new voice prompt

Install `edge-tts` and `ffmpeg`, activate the project's Python environment,
then generate an MP3 and convert it to the mono 16 kHz PCM WAV format used by
the agent:

```bash
python -m pip install edge-tts

edge-tts \
  --voice pl-PL-ZofiaNeural \
  --text "Your new Polish prompt" \
  --write-media /tmp/prompt.mp3

ffmpeg -y -loglevel error \
  -i /tmp/prompt.mp3 \
  -ar 16000 -ac 1 -c:a pcm_s16le \
  audio/prompt.wav
```

Use the resulting file from the `audio/` directory in the dialog code. Replace
an existing prompt only when its filename and purpose match the dialog step.

## License

This project is distributed under GPL-2.0-or-later because it uses PJSIP,
which is available under GPL or a separate proprietary license. See `LICENSE`
and `THIRD_PARTY_NOTICES.md`.
