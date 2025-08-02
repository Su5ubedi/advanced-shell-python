#!/usr/bin/env python3
"""
types.py - Data types and enums for Advanced Shell Simulation
"""

import subprocess
import time
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum


class JobStatus(Enum):
    """Job status enumeration"""
    RUNNING = "Running"
    STOPPED = "Stopped"
    DONE = "Done"


@dataclass
class Job:
    """Represents a background job"""
    id: int
    pid: int
    command: str
    args: List[str]
    status: JobStatus
    process: subprocess.Popen
    start_time: float
    end_time: Optional[float] = None
    background: bool = True

    def is_alive(self) -> bool:
        """Check if the process is still running"""
        if self.process.poll() is None:
            return True
        else:
            if self.status != JobStatus.DONE:
                self.status = JobStatus.DONE
                self.end_time = time.time()
            return False


@dataclass
class ParsedCommand:
    """Represents a parsed command"""
    command: str
    args: List[str]
    background: bool = False
    pipes: List[List[str]] = field(default_factory=list)