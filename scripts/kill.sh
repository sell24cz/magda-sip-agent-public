#!/usr/bin/env bash

set -e

APP_DIR="/opt/asystent"
CURRENT_UID="$(id -u)"

pids="$(pgrep -u "$CURRENT_UID" -f '[p]ython(3)? .*agent\.py' || true)"
targets=""

for pid in $pids; do
    process_dir="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"

    if [ "$process_dir" = "$APP_DIR" ]; then
        targets="$targets $pid"
    fi
done

if [ -z "$targets" ]; then
    echo "[STOP] Magda SIP Agent nie jest uruchomiona"
    exit 0
fi

for pid in $targets; do
    echo "[STOP] Zatrzymuję agent.py (PID $pid)"
    kill -TERM "$pid" 2>/dev/null || true
done

# Czekaj maksymalnie 5 sekund na poprawne zakończenie.
for _ in 1 2 3 4 5; do
    still_running=false

    for pid in $targets; do
        if kill -0 "$pid" 2>/dev/null; then
            still_running=true
        fi
    done

    if [ "$still_running" = false ]; then
        echo "[STOP] Agent został zatrzymany"
        exit 0
    fi

    sleep 1
done

# Wymuś zakończenie tylko procesów, które nadal działają
# i nadal należą do aplikacji /opt/asystent.
for pid in $targets; do
    if kill -0 "$pid" 2>/dev/null; then
        process_dir="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"

        if [ "$process_dir" = "$APP_DIR" ]; then
            echo "[STOP] Wymuszam zakończenie PID $pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
done

echo "[STOP] Agent został zatrzymany"