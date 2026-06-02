"""
Logging utilities: emit to both console and file.
"""
import os
import sys
import re
from datetime import datetime
from typing import Optional
import skillnet.utils as U


class DualLogger:
    """Dual logger: writes to both console and file."""
    
    # ANSI color-code regex
    ANSI_COLOR_PATTERN = re.compile(r'\033\[[0-9;]*m')
    
    def __init__(self, log_file_path: str, component_name: str = "PSN"):
        """
        Initialize the dual logger.
        
        Args:
            log_file_path: log file path
            component_name: component name (used as the log prefix)
        """
        self.log_file_path = log_file_path
        self.component_name = component_name
        
        # Ensure the log directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            U.f_mkdir(log_dir)
        
        # Open the log file (append mode)
        self.log_file = open(log_file_path, 'a', encoding='utf-8')
    
    def _strip_ansi_codes(self, text: str) -> str:
        """Strip ANSI color codes."""
        return self.ANSI_COLOR_PATTERN.sub('', text)
    
    def _format_message(self, message: str, level: str = "INFO") -> str:
        """
        Format log message (strip color codes).
        
        Args:
            message: original message (may contain ANSI color codes)
            level: log level
        
        Returns:
            formatted message (no color codes)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Strip color codes
        clean_message = self._strip_ansi_codes(message)
        return f"[{timestamp}] [{level}] [{self.component_name}] {clean_message}\n"
    
    def _write_to_file(self, formatted_message: str):
        """Write to file."""
        try:
            self.log_file.write(formatted_message)
            self.log_file.flush()  # flush immediately so logs are written promptly
        except Exception as e:
            # If write fails, at least emit to stderr
            print(f"Failed to write to log file: {e}", file=sys.stderr)
    
    def log(self, message: str, level: str = "INFO"):
        """
        Generic log method.
        
        Args:
            message: log message (may contain ANSI color codes)
            level: log level
        """
        # Ensure the message ends with a newline (check after stripping ANSI)
        clean_message = self._strip_ansi_codes(message)
        if not clean_message.endswith('\n'):
            message = message + '\n'
        
        # Emit to console (preserve color codes)
        print(message, end='')
        
        # Write to file (strip color codes)
        formatted = self._format_message(message, level)
        self._write_to_file(formatted)
    
    def info(self, message: str):
        """Log an info message."""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """Log a warning."""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """Log an error."""
        self.log(message, "ERROR")
    
    def debug(self, message: str):
        """Log a debug message."""
        self.log(message, "DEBUG")
    
    def close(self):
        """Close the log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
    
    def __del__(self):
        """Destructor: ensure the file is closed."""
        self.close()


class Tee:
    """Tee: write to both console and file."""
    
    def __init__(self, file_path: str, stream):
        """
        Initialize Tee.
        
        Args:
            file_path: log file path
            stream: original stream (sys.stdout or sys.stderr)
        """
        self.file_path = file_path
        self.stream = stream
        self.file = None
        
        # Ensure the log directory exists
        log_dir = os.path.dirname(file_path)
        if log_dir:
            U.f_mkdir(log_dir)
        
        # Open the log file (append mode)
        self.file = open(file_path, 'a', encoding='utf-8')
    
    def write(self, data):
        """Write data to both console and file."""
        # Write to console (preserve original behavior)
        self.stream.write(data)
        self.stream.flush()
        
        # Write to file (strip ANSI color codes)
        if self.file:
            clean_data = DualLogger.ANSI_COLOR_PATTERN.sub('', data)
            self.file.write(clean_data)
            self.file.flush()
    
    def flush(self):
        """Flush the stream."""
        self.stream.flush()
        if self.file:
            self.file.flush()
    
    def close(self):
        """Close the file."""
        if self.file:
            self.file.close()
            self.file = None
    
    def __del__(self):
        """Destructor: ensure the file is closed."""
        self.close()


class GlobalOutputLogger:
    """Global output logger: capture all stdout and stderr output."""
    
    _instance = None
    _stdout_tee = None
    _stderr_tee = None
    _original_stdout = None
    _original_stderr = None
    
    def __init__(self, ckpt_dir: str = "ckpt"):
        """
        Initialize the global output logger.
        
        Args:
            ckpt_dir: checkpoint directory
        """
        if GlobalOutputLogger._instance is not None:
            return
        
        self.ckpt_dir = ckpt_dir
        self.log_dir = U.f_join(ckpt_dir, "logs")
        U.f_mkdir(self.log_dir)
        
        # Build log file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stdout_log_file = U.f_join(self.log_dir, f"stdout_{timestamp}.log")
        stderr_log_file = U.f_join(self.log_dir, f"stderr_{timestamp}.log")
        
        # Save original streams (before redirection)
        GlobalOutputLogger._original_stdout = sys.stdout
        GlobalOutputLogger._original_stderr = sys.stderr
        
        # Create Tee objects (using the saved original streams)
        GlobalOutputLogger._stdout_tee = Tee(stdout_log_file, GlobalOutputLogger._original_stdout)
        GlobalOutputLogger._stderr_tee = Tee(stderr_log_file, GlobalOutputLogger._original_stderr)
        
        # Redirect stdout and stderr
        sys.stdout = GlobalOutputLogger._stdout_tee
        sys.stderr = GlobalOutputLogger._stderr_tee
        
        GlobalOutputLogger._instance = self
        
        # Print initialization info
        print(f"\033[36m[Global Logger] Global logging started\033[0m")
        print(f"\033[36m[Global Logger] stdout log: {stdout_log_file}\033[0m")
        print(f"\033[36m[Global Logger] stderr log: {stderr_log_file}\033[0m")
    
    def close(self):
        """Close the logger and restore original streams."""
        if GlobalOutputLogger._stdout_tee:
            sys.stdout = GlobalOutputLogger._original_stdout
            GlobalOutputLogger._stdout_tee.close()
            GlobalOutputLogger._stdout_tee = None
        
        if GlobalOutputLogger._stderr_tee:
            sys.stderr = GlobalOutputLogger._original_stderr
            GlobalOutputLogger._stderr_tee.close()
            GlobalOutputLogger._stderr_tee = None
        
        GlobalOutputLogger._instance = None
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        return cls._instance


class LoggerManager:
    """Log manager: manage multiple loggers."""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, component_name: str, ckpt_dir: str = "ckpt") -> DualLogger:
        """
        Get or create a logger.
        
        Args:
            component_name: component name (e.g. "GraphPlanner", "SkillGraphManager")
            ckpt_dir: checkpoint directory
        
        Returns:
            DualLogger instance
        """
        if component_name not in cls._loggers:
            # Build log file paths
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = U.f_join(ckpt_dir, "logs")
            log_file = U.f_join(log_dir, f"{component_name}_{timestamp}.log")
            
            cls._loggers[component_name] = DualLogger(log_file, component_name)
        
        return cls._loggers[component_name]
    
    @classmethod
    def close_all(cls):
        """Close all loggers."""
        for logger in cls._loggers.values():
            logger.close()
        cls._loggers.clear()

