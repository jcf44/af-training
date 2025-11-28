import asyncio
import logging
from typing import Dict, Optional
import subprocess
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessManager:
    _instance = None
    _processes: Dict[int, subprocess.Popen] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProcessManager, cls).__new__(cls)
        return cls._instance

    def start_process(self, command: list, log_file_path: str) -> int:
        """
        Starts a subprocess and logs its output to a file.
        Returns the PID of the process.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            
            with open(log_file_path, "w") as log_file:
                process = subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.getcwd() # Run from current directory (project root)
                )
            
            self._processes[process.pid] = process
            logger.info(f"Started process {process.pid} with command: {' '.join(command)}")
            return process.pid
        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            raise

    def stop_process(self, pid: int) -> bool:
        """
        Stops a running process by PID.
        """
        process = self._processes.get(pid)
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            
            del self._processes[pid]
            logger.info(f"Stopped process {pid}")
            return True
        return False

    def is_process_running(self, pid: int) -> bool:
        """
        Checks if a process is still running.
        """
        process = self._processes.get(pid)
        if process:
            return process.poll() is None
        return False

process_manager = ProcessManager()
