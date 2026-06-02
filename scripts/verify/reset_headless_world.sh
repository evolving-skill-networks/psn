#!/usr/bin/env bash
# Reset and start a fresh headless Fabric server (default: <repo-parent>/mc_replay_server/).
#
# Stops any running java listener on the configured port (default 25565),
# archives the existing world/ and logs/ with a timestamped suffix
# (so prior bot state can't leak into the next run), starts the server
# in the background, and waits up to 120s for the "Done (X.XXXs)!" line.
#
# Usage:
#   bash scripts/verify/reset_headless_world.sh <suffix>
#
# Example: bash scripts/verify/reset_headless_world.sh phase5
#
# Exits 0 once the server reports Done; exit 1 on timeout or kill failure.
# Prints the server PID to stdout (capture with $(...)).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"

SUFFIX="${1:-manual}"
# Headless server lives alongside the repo by default; override with MC_SERVER_DIR.
SERVER_DIR="${MC_SERVER_DIR:-$(dirname "$REPO")/mc_replay_server}"
PORT="${MC_HEADLESS_PORT:-25565}"
TS="$(date +%Y%m%d_%H%M%S)"

cd "$SERVER_DIR"

# 1. Stop existing listener on PORT
existing_pid="$(ss -lntpH "sport = :${PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
if [[ -n "$existing_pid" ]]; then
    echo "[reset] Killing existing server pid=$existing_pid on port $PORT" >&2
    kill "$existing_pid" 2>/dev/null || true
    for i in $(seq 1 30); do
        kill -0 "$existing_pid" 2>/dev/null || break
        sleep 1
    done
    kill -0 "$existing_pid" 2>/dev/null && kill -9 "$existing_pid"
fi

# 2. Archive current world (if any)
if [[ -d "world" ]]; then
    mv world "world.archived_${TS}_pre_verify_${SUFFIX}"
    echo "[reset] Archived world → world.archived_${TS}_pre_verify_${SUFFIX}" >&2
fi

# 3. Archive current logs (if any)
if [[ -d "logs" ]]; then
    mv logs "logs.archived_${TS}_pre_verify_${SUFFIX}"
    echo "[reset] Archived logs → logs.archived_${TS}_pre_verify_${SUFFIX}" >&2
fi

# 4. Start server in background
nohup java -jar fabric-server-launcher.jar nogui > "mc_server_${TS}.log" 2>&1 &
SERVER_PID=$!
echo "[reset] Started server pid=$SERVER_PID, log=mc_server_${TS}.log" >&2

# 5. Wait for "Done (X.XXXs)!"
for i in $(seq 1 120); do
    if grep -q 'Done (' "mc_server_${TS}.log" 2>/dev/null; then
        echo "[reset] Server READY in ${i}s" >&2
        echo "$SERVER_PID"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[reset] ERROR: server pid=$SERVER_PID died during boot. Tail:" >&2
        tail -20 "mc_server_${TS}.log" >&2
        exit 1
    fi
    sleep 1
done

echo "[reset] ERROR: server did not report Done within 120s. Tail:" >&2
tail -20 "mc_server_${TS}.log" >&2
exit 1
