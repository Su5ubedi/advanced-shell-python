#!/usr/bin/env python3
"""
job_manager.py - Job control operations for Advanced Shell Simulation
"""
import os
import signal
import subprocess
import time
from typing import Dict, List, Optional

from shell_types import Job, JobStatus
from scheduler import Scheduler, SchedulingAlgorithm


class JobManager:
    """Handles job control operations"""

    def __init__(self):
        self.jobs: Dict[int, Job] = {}
        self.job_counter = 0
        self.scheduler = Scheduler()
        
        # Set up scheduler callback
        self.scheduler.on_process_complete = self._on_scheduled_job_complete

    def add_job(self, command: str, args: List[str], process: subprocess.Popen, background: bool = True) -> Job:
        """Add a new job to the manager"""
        self.job_counter += 1
        job = Job(
            id=self.job_counter,
            pid=process.pid,
            command=command,
            args=args,
            status=JobStatus.RUNNING,
            process=process,
            start_time=time.time(),
            background=background
        )
        self.jobs[self.job_counter] = job
        return job

    def add_scheduled_job(self, command: str, args: List[str], priority: int = 5, 
                         time_needed: float = 5.0, background: bool = True) -> Job:
        """Add a new job to the scheduler"""
        self.job_counter += 1
        
        # Create a dummy process for scheduled jobs
        class DummyProcess:
            def __init__(self, job_id):
                self.pid = job_id
                self._poll_result = None
            
            def poll(self):
                return self._poll_result
            
            def terminate(self):
                pass
            
            def kill(self):
                pass
        
        dummy_process = DummyProcess(self.job_counter)
        
        job = Job(
            id=self.job_counter,
            pid=self.job_counter,  # Use job ID as PID for scheduled jobs
            command=command,
            args=args,
            status=JobStatus.WAITING,
            process=dummy_process,
            start_time=time.time(),
            background=background,
            priority=priority,
            total_time_needed=time_needed
        )
        
        self.jobs[self.job_counter] = job
        
        # Add to scheduler
        self.scheduler.add_process(job, priority, time_needed)
        
        return job

    def _on_scheduled_job_complete(self, job: Job):
        """Callback when a scheduled job completes"""
        job.status = JobStatus.DONE
        job.end_time = time.time()
        print(f"[{job.id}]+ Done\t\t{job.command}")

    def set_scheduling_algorithm(self, algorithm: str, time_slice: float = 2.0):
        """Set the scheduling algorithm"""
        try:
            if algorithm.lower() == "round_robin":
                self.scheduler.set_algorithm(SchedulingAlgorithm.ROUND_ROBIN, time_slice)
            elif algorithm.lower() == "priority":
                self.scheduler.set_algorithm(SchedulingAlgorithm.PRIORITY)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        except Exception as e:
            raise ValueError(f"Failed to set scheduling algorithm: {e}")

    def get_scheduler_status(self) -> dict:
        """Get scheduler status"""
        return self.scheduler.get_scheduler_status()

    def get_job(self, job_id: int) -> Optional[Job]:
        """Retrieve a job by ID"""
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> List[Job]:
        """Return all jobs"""
        return list(self.jobs.values())

    def list_jobs(self):
        """List all jobs with their status"""
        if not self.jobs:
            print("No active jobs")
            return

        print("Active jobs:")
        for job in self.jobs.values():
            job.is_alive()  # Update status for non-scheduled jobs
            duration = time.time() - job.start_time
            if job.end_time:
                duration = job.end_time - job.start_time

            # Show additional info for scheduled jobs
            if hasattr(job, 'priority') and job.priority:
                status_info = f"{job.status.value} (Priority: {job.priority})"
                if job.total_time_needed > 0:
                    status_info += f", Time: {job.execution_time:.1f}/{job.total_time_needed:.1f}s"
            else:
                status_info = job.status.value

            print(f"[{job.id}] {status_info} {job.command} (PID: {job.pid}, Duration: {int(duration)}s)")

    def bring_to_foreground(self, job_id: int) -> bool:
        """Bring a background job to the foreground"""
        job = self.get_job(job_id)
        if not job:
            print(f"fg: job {job_id} not found")
            return False

        if job.status == JobStatus.DONE:
            print(f"fg: job {job_id} has already completed")
            return False

        print(f"Bringing job [{job.id}] to foreground: {job.command}")

        # Handle scheduled jobs differently
        if hasattr(job, 'priority') and job.priority:
            print(f"Note: This is a scheduled job (Priority: {job.priority})")
            print("Scheduled jobs run automatically based on the scheduling algorithm")
            return True

        try:
            # Resume the process if it's stopped
            if job.status == JobStatus.STOPPED:
                os.kill(job.pid, signal.SIGCONT)

            job.status = JobStatus.RUNNING
            job.background = False

            # Wait for the job to complete in foreground
            try:
                job.process.wait()
                print(f"Job [{job.id}] completed")
            except KeyboardInterrupt:
                print(f"\nJob [{job.id}] interrupted")
                job.process.terminate()
                try:
                    job.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    job.process.kill()

            job.status = JobStatus.DONE
            job.end_time = time.time()

            # Remove from jobs list since it's completed
            del self.jobs[job_id]
            return True

        except ProcessLookupError:
            print(f"fg: process {job.pid} no longer exists")
            job.status = JobStatus.DONE
            if job_id in self.jobs:
                del self.jobs[job_id]
            return False
        except Exception as e:
            print(f"fg: failed to bring job to foreground: {e}")
            return False

    def resume_in_background(self, job_id: int) -> bool:
        """Resume a stopped job in the background"""
        job = self.get_job(job_id)
        if not job:
            print(f"bg: job {job_id} not found")
            return False

        if job.status == JobStatus.DONE:
            print(f"bg: job {job_id} has already completed")
            return False

        if job.status != JobStatus.STOPPED:
            print(f"bg: job {job_id} is not stopped")
            return False

        print(f"Resuming job [{job.id}] in background: {job.command}")

        try:
            os.kill(job.pid, signal.SIGCONT)
            job.status = JobStatus.RUNNING
            job.background = True
            return True
        except ProcessLookupError:
            print(f"bg: process {job.pid} no longer exists")
            job.status = JobStatus.DONE
            if job_id in self.jobs:
                del self.jobs[job_id]
            return False
        except Exception as e:
            print(f"bg: failed to resume job: {e}")
            return False

    def kill_job(self, job_id: int) -> bool:
        """Kill a job by sending SIGTERM"""
        job = self.get_job(job_id)
        if not job:
            return False

        if job.status == JobStatus.DONE:
            return False

        print(f"Terminating job [{job.id}]: {job.command}")

        # Handle scheduled jobs
        if hasattr(job, 'priority') and job.priority:
            self.scheduler.remove_process(job_id)
            job.status = JobStatus.DONE
            job.end_time = time.time()
            return True

        try:
            job.process.terminate()
            try:
                job.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                job.process.kill()

            job.status = JobStatus.DONE
            job.end_time = time.time()
            return True
        except Exception as e:
            print(f"Failed to kill job: {e}")
            return False

    def cleanup_completed_jobs(self):
        """Remove completed jobs from the manager"""
        completed_jobs = []
        for job_id, job in list(self.jobs.items()):
            # For non-scheduled jobs, check if alive
            if not hasattr(job, 'priority') or not job.priority:
                job.is_alive()  # Update status
            
            if job.status == JobStatus.DONE:
                completed_jobs.append(job_id)
                print(f"[{job_id}]+ Done\t\t{job.command}")

        for job_id in completed_jobs:
            if job_id in self.jobs:
                del self.jobs[job_id]