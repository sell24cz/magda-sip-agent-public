#!/usr/bin/env bash

set -e

APP_DIR="/opt/asystent"
cd "$APP_DIR"

# Lokalne dane konta nie są przechowywane w kodzie ani w Git.
if [ -f "$APP_DIR/secrets/sip.env" ]; then
    source "$APP_DIR/secrets/sip.env"
fi

# Opcjonalne zmienne środowiskowe z profilu użytkownika.
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
fi

for variable_name in SIP_USER SIP_SERVER SIP_PASSWORD; do
    if [ -z "${!variable_name:-}" ]; then
        echo "[START ERROR] Brak zmiennej $variable_name"
        exit 1
    fi
done

source "$APP_DIR/venv/bin/activate"

# Log zawiera dane rozmów, dlatego jest dostępny tylko dla właściciela.
umask 077
mkdir -p "$APP_DIR/logs"
LOG_FILE="$APP_DIR/logs/agent_$(date +%F).log"

# Konsola zachowuje kolory, a plik otrzymuje czysty tekst bez kodów ANSI.
exec > >(tee >(sed -u 's/\x1B\[[0-9;]*[mK]//g' >> "$LOG_FILE")) 2>&1

echo
echo "=================================================="
echo "[START] $(date '+%F %T')"
echo "[START] Log: $LOG_FILE"

# Najpierw sprawdź nowy kod. Działający agent pozostaje aktywny,
# jeśli nowa wersja zawiera błąd składni.
python3 -m py_compile "$APP_DIR/agent.py"

echo "[START] Składnia agent.py: OK"

# Zamknij poprzedni agent.py tego samego użytkownika, ale wyłącznie
# gdy proces pracuje z katalogu /opt/asystent.
pids="$(pgrep -u "$(id -u)" -f '[p]ython(3)? .*agent\.py' || true)"

for pid in $pids; do
    process_dir="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"

    if [ "$process_dir" = "$APP_DIR" ]; then
        echo "[START] Zatrzymuję poprzedni agent.py (PID $pid)"
        kill -TERM "$pid" 2>/dev/null || true
    fi
done

# Daj procesowi maksymalnie 5 sekund na poprawne zakończenie.
for _ in 1 2 3 4 5; do
    still_running=false

    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            still_running=true
        fi
    done

    if [ "$still_running" = false ]; then
        break
    fi

    sleep 1
done

# Awaryjnie zakończ proces, który nie zareagował na SIGTERM.
for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
        process_dir="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"

        if [ "$process_dir" = "$APP_DIR" ]; then
            echo "[START] Wymuszam zakończenie PID $pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
done

echo "[START] Uruchamiam Magda SIP Agent"
exec python3 -u "$APP_DIR/agent.py"
