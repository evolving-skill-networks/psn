import time
import re
import warnings
from typing import List

import psutil
import subprocess
import logging
import threading

import skillnet.utils as U


class SubprocessMonitor:
    def __init__(
        self,
        commands: List[str],
        name: str,
        ready_match: str = r".*",
        log_path: str = "logs",
        callback_match: str = r"^(?!x)x$",  # regex that will never match
        callback: callable = None,
        finished_callback: callable = None,
    ):
        self.commands = commands
        start_time = time.strftime("%Y%m%d_%H%M%S")
        self.name = name
        self.logger = logging.getLogger(name)
        handler = logging.FileHandler(U.f_join(log_path, f"{start_time}.log"))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.process = None
        self.ready_match = ready_match
        self.ready_event = None
        self.ready_line = None
        self.callback_match = callback_match
        self.callback = callback
        self.finished_callback = finished_callback
        self.thread = None

    def _start(self):
        self.logger.info(f"Starting subprocess with commands: {self.commands}")

        self.process = psutil.Popen(
            self.commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        print(f"Subprocess {self.name} started with PID {self.process.pid}.")
        for line in iter(self.process.stdout.readline, ""):
            self.logger.info(line.strip())
            if re.search(self.ready_match, line):
                self.ready_line = line
                self.logger.info("Subprocess is ready.")
                self.ready_event.set()
            if re.search(self.callback_match, line):
                self.callback()
        if not self.ready_event.is_set():
            self.ready_event.set()
            warnings.warn(f"Subprocess {self.name} failed to start.")
        if self.finished_callback:
            self.finished_callback()

    def run(self):
        self.ready_event = threading.Event()
        self.ready_line = None
        # daemon=True so this stdout-tailing thread does NOT hold Python
        # alive at process exit. Without daemon, Python's "wait for non-daemon
        # threads" rule traps us indefinitely: the thread sits in
        # `for line in iter(self.process.stdout.readline, "")` waiting on a
        # readline that never returns until the subprocess closes its stdout
        # — which never happens unless someone calls self.stop() first.
        self.thread = threading.Thread(target=self._start, daemon=True)
        self.thread.start()
        self.ready_event.wait()

    def stop(self):
        self.logger.info("Stopping subprocess.")
        if self.process and self.process.is_running():
            # Kill all child processes to ensure ports are released
            try:
                children = self.process.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                # Wait for child processes to terminate
                psutil.wait_procs(children, timeout=3)
                # Force kill child processes still running
                for child in children:
                    try:
                        if child.is_running():
                            child.kill()
                    except psutil.NoSuchProcess:
                        pass
            except psutil.NoSuchProcess:
                pass

            # Terminate parent process
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except psutil.TimeoutExpired:
                self.logger.warning("Process did not terminate gracefully, killing...")
                self.process.kill()
                self.process.wait()

    # def __del__(self):
    # if self.process.is_running():
    # self.stop()

    @property
    def is_running(self):
        if self.process is None:
            return False
        return self.process.is_running()
