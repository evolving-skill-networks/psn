#!/bin/bash
# Start the headless Fabric server set up by setup_fabric_server.sh.
#
# Usage:
#     # Foreground (CTRL+C to stop):
#     bash installation/run_fabric_server.sh
#
#     # Background, returns port on stdout, PID file at .pid:
#     bash installation/run_fabric_server.sh --daemon
#
#     # Stop a daemonized server:
#     bash installation/run_fabric_server.sh --stop
#
# skillnet.sh invokes this with --daemon when launched with --mc-mode=headless,
# captures the port, runs PSN, and then calls --stop.

set -euo pipefail

PSN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$PSN_ROOT/installation/fabric_server"
CONFIG_FILE="$PSN_ROOT/installation/fabric_server_config.json"
PID_FILE="$SERVER_DIR/.pid"
LOG_FILE="$SERVER_DIR/.psn_run.log"

read_cfg() {
    python3 -c "import json; print(json.load(open('$CONFIG_FILE')).get('$1', ''))"
}

if [[ "${1:-}" == "--stop" ]]; then
    if [[ ! -f "$PID_FILE" ]]; then
        echo "[run_fabric_server] no running server (no $PID_FILE)" >&2
        exit 0
    fi
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" >/dev/null 2>&1; then
        echo "[run_fabric_server] stopping server (PID $PID) ..."
        # Graceful: send `stop` to the server's stdin via the FIFO if alive.
        if [[ -p "$SERVER_DIR/.stdin_fifo" ]]; then
            echo "stop" > "$SERVER_DIR/.stdin_fifo" || true
        fi
        # Wait up to 30s for graceful shutdown.
        for _ in $(seq 1 30); do
            ps -p "$PID" >/dev/null 2>&1 || break
            sleep 1
        done
        # Force-kill if still alive.
        ps -p "$PID" >/dev/null 2>&1 && kill -KILL "$PID" 2>/dev/null || true
    fi
    rm -f "$PID_FILE" "$SERVER_DIR/.stdin_fifo"
    echo "[run_fabric_server] stopped"
    exit 0
fi

DAEMON=false
if [[ "${1:-}" == "--daemon" ]]; then
    DAEMON=true
fi

# Pre-flight: ensure setup has been done.
if [[ ! -f "$SERVER_DIR/fabric-server-launcher.jar" ]]; then
    echo "ERROR: Fabric server not installed. Run:" >&2
    echo "    bash installation/setup_fabric_server.sh" >&2
    exit 1
fi
if [[ ! -f "$SERVER_DIR/server.properties" ]]; then
    echo "ERROR: server.properties missing. Re-run setup_fabric_server.sh." >&2
    exit 1
fi

MEM_MIN=$(read_cfg memory_min_mb)
MEM_MAX=$(read_cfg memory_max_mb)
PORT=$(read_cfg server_port)

# If a server is already running on PID_FILE, just print its port and exit.
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" >/dev/null 2>&1; then
        echo "[run_fabric_server] server already running (PID $OLD_PID, port $PORT)" >&2
        echo "$PORT"
        exit 0
    else
        rm -f "$PID_FILE"
    fi
fi

cd "$SERVER_DIR"

if $DAEMON; then
    # Use a FIFO for stdin so --stop can send `stop` command later.
    rm -f .stdin_fifo
    mkfifo .stdin_fifo

    # Open the FIFO for write on FD 3 in the BACKGROUND keeper process,
    # so the FIFO stays open until --stop closes it. The server's stdin
    # reads from the FIFO. Without the keeper FD, the FIFO closes as
    # soon as the writer exits.
    (
        # Keep FIFO open for write — minimal sleeper so it stays alive.
        exec 3>.stdin_fifo
        while ps -p "$$" >/dev/null 2>&1; do
            sleep 60
        done
    ) &
    FIFO_KEEPER_PID=$!

    # Launch the server in the background, reading stdin from the FIFO,
    # writing stdout/stderr to the log file. The server's exit propagates
    # via PID; FIFO keeper is reaped by parent (skillnet.sh) on --stop.
    nohup java -Xms${MEM_MIN}M -Xmx${MEM_MAX}M -jar fabric-server-launcher.jar nogui \
        < .stdin_fifo > "$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$PID_FILE"

    echo "[run_fabric_server] starting server (PID $SERVER_PID), waiting for 'Done' ..." >&2
    for i in $(seq 1 120); do
        if grep -q 'Done (' "$LOG_FILE" 2>/dev/null; then
            break
        fi
        if ! ps -p "$SERVER_PID" >/dev/null 2>&1; then
            echo "ERROR: server died during startup. Tail of log:" >&2
            tail -30 "$LOG_FILE" >&2
            rm -f "$PID_FILE"
            kill "$FIFO_KEEPER_PID" 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done

    if ! grep -q 'Done (' "$LOG_FILE" 2>/dev/null; then
        echo "ERROR: server did not finish startup within 120s. Tail of log:" >&2
        tail -30 "$LOG_FILE" >&2
        kill "$SERVER_PID" 2>/dev/null || true
        rm -f "$PID_FILE"
        kill "$FIFO_KEEPER_PID" 2>/dev/null || true
        exit 1
    fi

    # On --daemon, print the port on stdout so the caller can capture it.
    # All other log lines go to stderr above so they do not pollute stdout.
    echo "$PORT"
    exit 0
fi

# Foreground mode: just exec the server.
exec java -Xms${MEM_MIN}M -Xmx${MEM_MAX}M -jar fabric-server-launcher.jar nogui
