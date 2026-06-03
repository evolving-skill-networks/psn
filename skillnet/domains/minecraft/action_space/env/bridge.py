import os
import os.path
import time
import socket
from typing import SupportsFloat, Any, Tuple, Dict

import requests
import json

import gymnasium as gym
from gymnasium.core import ObsType

import skillnet.utils as U

from .process_monitor import SubprocessMonitor


class SkillNetEnv(gym.Env):
    def __init__(
        self,
        mc_port=None,
        server_host="http://127.0.0.1",
        server_port=3000,
        # Sits above the mineflayer /step wall-clock guardrail (240s, see
        # index.js PSN_STEP_WALLCLOCK_SEC) so the server self-aborts a wedged
        # step and replies with a 500 BEFORE this read-timeout fires. The
        # timeout is the backstop for the one case the guardrail can't catch:
        # a fully synchronous busy-loop that blocks node's event loop so its
        # own setTimeout never runs. Then this caps the dead-wait at 300s
        # (was 900s) and the next step() restarts the wedged process.
        request_timeout=300,
        log_path="./logs",
    ):
        if not mc_port:
            raise ValueError("mc_port must be specified")
        self.mc_port = mc_port
        self.server = f"{server_host}:{server_port}"
        self.server_port = server_port
        self.request_timeout = request_timeout
        self.log_path = log_path
        self.mineflayer = self.get_mineflayer_process(server_port)
        self.has_reset = False
        self.reset_options = None
        self.connected = False
        self.server_paused = False
        self.step_count = 0  # tally env step calls
        # Client-side mcpr recording (opt-in via PSN_MCPR_OUTPUT env var).
        # Records every server→client packet — including SetContainerSlot /
        # SetContainerContent which ServerReplay drops — so downstream
        # tooling can extract a frame-exact inventory timeline. Leave the
        # env var unset to opt out (default: disabled).
        self._mcpr_output = os.environ.get("PSN_MCPR_OUTPUT") or None
        self._mcpr_recording = False
        # Each subprocess restart spawns its own .mcpr fragment; we collect
        # them here so close() can merge into a single continuous mcpr.
        self._mcpr_fragments = []

    def get_mineflayer_process(self, server_port):
        U.f_mkdir(self.log_path, "mineflayer")
        file_path = os.path.abspath(os.path.dirname(__file__))
        return SubprocessMonitor(
            commands=[
                "node",
                U.f_join(file_path, "mineflayer/index.js"),
                str(server_port),
            ],
            name="mineflayer",
            ready_match=r"Server started on port (\d+)",
            log_path=U.f_join(self.log_path, "mineflayer"),
        )

    def check_process(self):
        # Ensure reset_options is initialized
        if self.reset_options is None:
            # If reset_options is unset, use defaults
            self.reset_options = {
                "port": self.mc_port,
                "reset": "hard",
                "inventory": {},
                "equipment": [],
                "spread": False,
                "waitTicks": 5,
                "position": None,
            }
        
        retry = 0
        while not self.mineflayer.is_running:
            print("Mineflayer process has exited, restarting")
            self.mineflayer.run()
            if not self.mineflayer.is_running:
                if retry >= 3:
                    raise RuntimeError("Mineflayer process failed to start")
                else:
                    retry += 1
                    time.sleep(1)  # wait, then retry
                    continue
            print(self.mineflayer.ready_line)
            
            # Wait for the server to fully start
            time.sleep(1)
            
            # Verify the Minecraft server is reachable
            mc_port = self.reset_options.get("port")
            if mc_port:
                # print(f"\033[36m[Check Process] Verifying Minecraft server localhost:{mc_port} reachable...\033[0m")
                server_ready = False
                for check_retry in range(5):
                    try:
                        sock = socket.create_connection(("localhost", mc_port), timeout=2)
                        sock.close()
                        server_ready = True
                        # print(f"\033[32m[Check Process] Minecraft server localhost:{mc_port} reachable\033[0m")
                        break
                    except (socket.error, OSError) as e:
                        if check_retry < 4:
                            # print(f"\033[33m[Check Process] Minecraft server not ready (attempt {check_retry + 1}/5); retrying after a wait...\033[0m")
                            time.sleep(2)
                        else:
                            print(f"\033[31m[Check Process] Warning: cannot connect to Minecraft server localhost:{mc_port}: {e}\033[0m")
                            print(f"\033[31m[Check Process] Please make sure:\033[0m")
                            print(f"\033[31m[Check Process]   1. The Minecraft server is running\033[0m")
                            print(f"\033[31m[Check Process]   2. LAN game is open (Open to LAN)\033[0m")
                            print(f"\033[31m[Check Process]   3. The server is not paused (press F3+P to unpause)\033[0m")
            
            # Add retry mechanism to handle connection errors
            start_retry = 0
            max_start_retries = 3
            while start_retry < max_start_retries:
                try:
                    # print(f"\033[36m[Check Process] Sending /start request to mineflayer (attempt {start_retry + 1}/{max_start_retries})...\033[0m")
                    # print(f"\033[36m[Check Process] Request params: port={self.reset_options.get('port')}, reset={self.reset_options.get('reset')}\033[0m")
                    res = requests.post(
                        f"{self.server}/start",
                        json=self.reset_options,
                        timeout=self.request_timeout,
                    )
                    if res.status_code != 200:
                        # Try fetching error details
                        error_detail = "Unknown error"
                        suggestions = []
                        try:
                            error_data = res.json()
                            error_detail = error_data.get("error", str(error_data))
                            suggestions = error_data.get("suggestions", [])
                        except:
                            error_detail = res.text[:500] if res.text else "Could not parse error response"
                        
                        error_msg = f"Minecraft server reply with code {res.status_code}: {error_detail}"
                        if suggestions:
                            error_msg += "\nSuggestions:\n" + "\n".join(f"  - {s}" for s in suggestions)
                        else:
                            error_msg += "\nSuggestions:\n  - Ensure the Minecraft server is running\n  - Ensure LAN game is open (Open to LAN)\n  - Ensure the server is not paused (press F3+P to unpause)"
                        
                        self.mineflayer.stop()
                        raise RuntimeError(error_msg)
                    # Kick off client-side mcpr recording if requested via
                    # PSN_MCPR_OUTPUT — opt-in, idempotent.
                    self._maybe_start_mcpr()
                    return res.json()
                except (requests.exceptions.ConnectionError, 
                        requests.exceptions.Timeout,
                        requests.exceptions.RequestException) as e:
                    start_retry += 1
                    if start_retry < max_start_retries:
                        # print(f"\033[33m[Check Process] Start request failed (attempt {start_retry}/{max_start_retries}): {e}\033[0m")
                        time.sleep(2)  # wait, then retry
                    else:
                        self.mineflayer.stop()
                        raise RuntimeError(
                            f"Failed to start mineflayer after {max_start_retries} connection attempts: {e}"
                        ) from e

    def check_http_server_health(self):
        """Check whether the HTTP server is responsive."""
        try:
            response = requests.get(
                f"{self.server}/health",
                timeout=2
            )
            return response.status_code == 200
        except:
            return False

    def step(
        self,
        code: str,
        programs: str = "",
        skill_names: list = None,
        is_iteration: bool = True,
        keep_paused: bool = False,
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if not self.has_reset:
            raise RuntimeError("Environment has not been reset yet")
        self.check_process()
        
        # Check HTTP server health
        if not self.check_http_server_health():
            print("\033[33m[Env Step] HTTP server unresponsive; checking process status...\033[0m")
            if not self.mineflayer.is_running:
                print("\033[33m[Env Step] Mineflayer process has exited; restarting...\033[0m")
                self.check_process()
                time.sleep(2)  # wait for the server to start
            else:
                print("\033[33m[Env Step] Process is running but HTTP server is unresponsive; retrying after a wait...\033[0m")
                time.sleep(2)
        
        # keep_paused: world-setup commands apply while the server is paused, so
        # the reset path runs them without unpausing (no idle window).
        if not keep_paused:
            self.unpause()
        data = {
            "code": code,
            "programs": programs,
            "keep_paused": keep_paused,
        }
        # Opt-in: dump every /step request body to disk so a failing skill
        # can be replayed byte-identically against a debug mineflayer.
        # Enabled by setting PSN_STEP_DUMP_DIR=<path>. Off by default.
        _step_dump_dir = os.environ.get("PSN_STEP_DUMP_DIR")
        if _step_dump_dir:
            try:
                os.makedirs(_step_dump_dir, exist_ok=True)
                _dump_payload = dict(data)
                if skill_names is not None:
                    _dump_payload["skill_names"] = skill_names
                _dump_ts = int(time.time() * 1000)
                _dump_path = os.path.join(
                    _step_dump_dir, f"req_{_dump_ts}_{self.step_count}.json"
                )
                with open(_dump_path, "w") as _f:
                    json.dump(_dump_payload, _f)
            except Exception as _dump_err:
                print(f"\033[33m[Env Step] step-dump failed: {_dump_err}\033[0m")
        # If skill_names was provided, add it to the request payload
        if skill_names is not None:
            data["skill_names"] = skill_names
        
        # Count only real task-execution iterations. Non-iteration callers
        # (revert-placed-blocks, reset peeks, observe refreshes) still hit
        # the HTTP /step endpoint but are not iterations of the agent loop.
        if is_iteration:
            self.step_count += 1
        
        # Add retry mechanism to handle connection errors
        max_retries = 3
        retry_count = 0
        last_exception = None
        
        while retry_count < max_retries:
            try:
                res = requests.post(
                    f"{self.server}/step", json=data, timeout=self.request_timeout
                )
                if res.status_code != 200:
                    raise RuntimeError(f"Failed to step Minecraft server: status code {res.status_code}")
                returned_data = res.json()
                if not keep_paused:
                    self.pause()
                
                # Emit iteration-counter log only for real iterations.
                if is_iteration:
                    print(f"\033[36m[Env Step] Iteration {self.step_count}\033[0m")
                
                # Mineflayer returns events as a parsed array; defensively
                # handle the legacy JSON-string shape as well.
                if isinstance(returned_data, str):
                    return json.loads(returned_data)
                return returned_data
                
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.RequestException) as e:
                last_exception = e
                retry_count += 1
                
                # Check whether it is a connection-interruption error
                error_str = str(e).lower()
                is_connection_error = (
                    "connection aborted" in error_str or
                    "remotedisconnected" in error_str or
                    "connection refused" in error_str or
                    "broken pipe" in error_str
                )
                # Define specific connection-error types
                is_connection_refused = "connection refused" in error_str
                is_connection_aborted = (
                    "connection aborted" in error_str or
                    "remotedisconnected" in error_str
                )
                
                if is_connection_error and retry_count < max_retries:
                    print(f"\033[33m[Env Step] Connection error (attempt {retry_count}/{max_retries}): {e}\033[0m")
                    # Check HTTP server health
                    http_healthy = self.check_http_server_health()
                    # Check and restart the mineflayer process
                    if not self.mineflayer.is_running:
                        print("\033[33m[Env Step] Mineflayer process has exited; restarting...\033[0m")
                        try:
                            self.check_process()
                            # Wait for the process to stabilize
                            time.sleep(3)  # increase wait time to ensure the process fully starts
                        except Exception as restart_error:
                            print(f"\033[31m[Env Step] Process restart failed: {restart_error}\033[0m")
                            # Restart failed; raise the exception
                            raise RuntimeError(f"Failed to restart mineflayer: {restart_error}") from e
                    elif is_connection_refused:
                        # Connection refused usually means the server crashed or never started
                        print("\033[33m[Env Step] Connection refused; server may have crashed; checking process status...\033[0m")
                        if not http_healthy:
                            print("\033[33m[Env Step] HTTP server unresponsive; attempting process restart...\033[0m")
                            try:
                                self.mineflayer.stop()
                                time.sleep(1)
                                self.check_process()
                                time.sleep(3)  # wait for restart to complete
                            except Exception as restart_error:
                                print(f"\033[31m[Env Step] Process restart failed: {restart_error}\033[0m")
                        else:
                            # HTTP healthy but connection refused; may be transient, wait longer
                            print("\033[33m[Env Step] HTTP server healthy but connection refused; waiting for recovery...\033[0m")
                            time.sleep(5)
                    elif is_connection_aborted:
                        # Connection aborted may indicate code is executing and the HTTP connection was closed
                        # In this case, wait longer to let the code finish
                        print("\033[33m[Env Step] Connection interrupted; code may be executing; waiting for completion...\033[0m")
                        if not http_healthy:
                            # HTTP server unresponsive; wait for recovery
                            print("\033[33m[Env Step] HTTP server unresponsive; waiting for recovery...\033[0m")
                            time.sleep(5)  # increase wait time
                        else:
                            # HTTP server healthy; code may be running long; wait longer
                            print("\033[33m[Env Step] HTTP server healthy; waiting for code execution to complete...\033[0m")
                            time.sleep(5)  # increase wait time to give code more time
                    elif not http_healthy:
                        # Process running but HTTP server unresponsive; may be recovering from a crash
                        print("\033[33m[Env Step] HTTP server unresponsive; waiting for recovery...\033[0m")
                        time.sleep(5)  # increase wait time
                    else:
                        # Process still running but connection failed; wait briefly before retrying
                        print("\033[33m[Env Step] Process running but connection failed; retrying after a wait...\033[0m")
                        time.sleep(3)  # increase wait time
                else:
                    # Non-connection error or max retries reached
                    raise RuntimeError(f"Failed to step Minecraft server after {retry_count} attempts: {e}") from e
        
        # If all retries failed
        raise RuntimeError(f"Failed to step Minecraft server after {max_retries} attempts: {last_exception}") from last_exception

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        if options is None:
            options = {}

        if options.get("inventory", {}) and options.get("mode", "hard") != "hard":
            raise RuntimeError("inventory can only be set when options is hard")

        self.reset_options = {
            "port": self.mc_port,
            "reset": options.get("mode", "hard"),
            "inventory": options.get("inventory", {}),
            "equipment": options.get("equipment", []),
            "spread": options.get("spread", False),
            "waitTicks": options.get("wait_ticks", 5),
            "position": options.get("position", None),
        }

        self.unpause()
        self.mineflayer.stop()
        time.sleep(2)  # wait for mineflayer to exit and port to be released

        returned_data = self.check_process()
        self.has_reset = True
        self.connected = True
        # All the reset in step will be soft
        self.reset_options["reset"] = "soft"
        self.pause()
        
        # On reset, print the accumulated step count (if greater than 0)
        if self.step_count > 0:
            print(f"\033[33m[Env Reset] Resetting environment; {self.step_count} iterations accumulated\033[0m")
            # Optional: reset the counter (if you want a fresh count after each reset)
            # self.step_count = 0
        
        return returned_data

    def _maybe_start_mcpr(self):
        """Start client-side mcpr recording if PSN_MCPR_OUTPUT is set.
        Called after EVERY /start success — PSN restarts the mineflayer
        subprocess between iterations (via env.reset(hard)) which kills
        the previous JS-side McprRecorder, so each subprocess lifetime
        gets its own .mcpr file (uniquified via {ts})."""
        if not self._mcpr_output:
            return
        # Always reset the flag before re-attempting — the previous
        # recorder lived in the previous (now-dead) subprocess.
        self._mcpr_recording = False
        out = os.path.abspath(os.path.expanduser(
            self._mcpr_output.replace("{ts}", str(int(time.time() * 1000)))))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            res = requests.post(
                f"{self.server}/start_mcpr",
                json={"output": out},
                timeout=10,
            )
            if res.status_code == 200:
                self._mcpr_recording = True
                self._mcpr_output_resolved = out
                # Track every fragment so close() can merge them.
                self._mcpr_fragments.append(out)
                print(f"\033[36m[mcpr] recording started → {out}\033[0m")
            else:
                print(f"\033[33m[mcpr] /start_mcpr failed ({res.status_code}): "
                      f"{res.text[:200]}\033[0m")
        except Exception as e:
            print(f"\033[33m[mcpr] /start_mcpr exception: {e}\033[0m")

    def _maybe_stop_mcpr(self):
        """Finalize the mcpr if we started one. Best-effort; ignore errors."""
        if not self._mcpr_recording:
            return
        try:
            res = requests.post(f"{self.server}/stop_mcpr", timeout=30)
            if res.status_code == 200:
                print(f"\033[32m[mcpr] recording stopped → "
                      f"{getattr(self, '_mcpr_output_resolved', '?')}\033[0m")
            else:
                print(f"\033[33m[mcpr] /stop_mcpr returned {res.status_code}: "
                      f"{res.text[:200]}\033[0m")
        except Exception as e:
            print(f"\033[33m[mcpr] /stop_mcpr exception: {e}\033[0m")
        finally:
            self._mcpr_recording = False

    def _maybe_merge_mcpr_fragments(self):
        """If we collected multiple .mcpr fragments during the run (one per
        subprocess lifetime), invoke scripts/merge_mcpr_run.py to splice
        them into a single continuous mcpr representing the whole PSN run.

        Output filename: <first_fragment_dir>/merged_run_<startts>.mcpr
        """
        # Filter to fragments that actually exist on disk (the SIGTERM
        # finalize-on-shutdown writes them; tmpdirs that didn't get
        # finalized for any reason show up here too — drop those).
        existing = [p for p in self._mcpr_fragments if os.path.isfile(p)]
        if len(existing) < 2:
            return  # nothing to merge
        first = existing[0]
        # Use the first fragment's epoch-ms in its filename as the merged
        # file's identifier so it's clear which run it belongs to.
        import re
        m = re.search(r"(\d{12,15})", os.path.basename(first))
        run_id = m.group(1) if m else str(int(time.time() * 1000))
        out_dir = os.path.dirname(first)
        merged_path = os.path.join(out_dir, f"merged_run_{run_id}.mcpr")
        # Invoke the merger script as a subprocess so its output streams
        # to console (helpful for the user to see what got merged).
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ".."))
        merger = os.path.join(repo_root, "scripts", "merge_mcpr_run.py")
        if not os.path.isfile(merger):
            print(f"\033[33m[mcpr] merger not found at {merger}; "
                  f"skipping (fragments: {existing})\033[0m")
            return
        import subprocess, sys as _sys
        try:
            cmd = [_sys.executable, merger] + existing + ["-o", merged_path]
            print(f"\033[36m[mcpr] merging {len(existing)} fragments → "
                  f"{merged_path}\033[0m")
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode == 0:
                print(f"\033[32m[mcpr] merged → {merged_path}\033[0m")
                if res.stdout.strip():
                    print(res.stdout)
            else:
                print(f"\033[33m[mcpr] merger failed (rc={res.returncode}): "
                      f"{res.stderr[:300]}\033[0m")
        except Exception as e:
            print(f"\033[33m[mcpr] merger exception: {e}\033[0m")

    def close(self):
        self.unpause()
        # Finalize the mcpr file BEFORE we tell mineflayer to /stop, so the
        # bot's _client (where the packet listener is attached) is still
        # alive and the recorder can flush in-flight packets cleanly.
        self._maybe_stop_mcpr()
        # Concatenate all per-subprocess fragments into one continuous mcpr.
        self._maybe_merge_mcpr_fragments()
        if self.connected:
            res = requests.post(f"{self.server}/stop")
            if res.status_code == 200:
                self.connected = False
        self.mineflayer.stop()
        
        # Print final step statistics
        if self.step_count > 0:
            print(f"\033[32m[Env Close] Environment closed; total iterations: {self.step_count}\033[0m")
        
        return not self.connected

    def pause(self):
        if self.mineflayer.is_running and not self.server_paused:
            try:
                res = requests.post(f"{self.server}/pause", timeout=10)
                if res.status_code == 200:
                    self.server_paused = True
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                # On connection error, check process status
                if not self.mineflayer.is_running:
                    print("\033[33m[Env Pause] Mineflayer process has exited; cannot pause\033[0m")
                else:
                    print(f"\033[33m[Env Pause] Pause request failed: {e}\033[0m")
        return self.server_paused

    def unpause(self):
        # Verify-and-retry semantics: the /pause endpoint is a toggle on the
        # MC side, and when MC is busy with the sync save that pause()
        # triggers, the chat command can be dropped if mineflayer dies
        # before the save drains. Pass expect_running=True so mineflayer
        # observes physicsTick events post-toggle and returns 503 if MC
        # didn't actually unpause. On 503 we PROBE via /observe_ticks
        # (read-only — no extra toggle) to check whether the chat landed
        # late; only retry the toggle if the probe also reports no ticks.
        if not (self.mineflayer.is_running and self.server_paused):
            return self.server_paused

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                res = requests.post(
                    f"{self.server}/pause",
                    json={"expect_running": True, "timeout_ms": 5000},
                    timeout=15,
                )
                if res.status_code == 200:
                    self.server_paused = False
                    return self.server_paused
                if res.status_code == 503:
                    print(
                        f"\033[33m[Env Unpause] attempt {attempt}/{max_attempts}: "
                        f"no ticks observed, waiting before retry\033[0m"
                    )
                    time.sleep(2)
                    try:
                        probe = requests.post(
                            f"{self.server}/observe_ticks",
                            json={"timeout_ms": 3000},
                            timeout=10,
                        )
                        if probe.status_code == 200 and probe.json().get("ticks", 0) >= 2:
                            self.server_paused = False
                            return self.server_paused
                    except (requests.exceptions.ConnectionError,
                            requests.exceptions.Timeout,
                            requests.exceptions.RequestException) as probe_err:
                        print(
                            f"\033[33m[Env Unpause] tick probe failed: "
                            f"{probe_err}; resending toggle\033[0m"
                        )
                    # Probe says still no ticks — original chat genuinely
                    # never landed. Loop and resend toggle.
                    continue
                print(f"\033[33m[Env Unpause] HTTP {res.status_code}: {res.text[:200]}\033[0m")
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                if not self.mineflayer.is_running:
                    print("\033[33m[Env Unpause] Mineflayer process has exited; treating as unpaused\033[0m")
                    self.server_paused = False
                    return self.server_paused
                print(f"\033[33m[Env Unpause] attempt {attempt}/{max_attempts} failed: {e}\033[0m")
        print(
            f"\033[31m[Env Unpause] WARNING: failed to confirm MC unpaused after "
            f"{max_attempts} attempts; subsequent reset may spawn-timeout\033[0m"
        )
        return self.server_paused
