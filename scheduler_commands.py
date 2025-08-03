#!/usr/bin/env python3
"""
scheduler_commands.py - Command handlers for process scheduling (Deliverable 2)
Enhanced with clear/reset functionality
"""

from typing import List
from process_scheduler import ProcessScheduler
from scheduler_types import SchedulingAlgorithm


class SchedulerCommands:
    """Command handlers for process scheduling functionality"""

    def __init__(self, scheduler: ProcessScheduler):
        self.scheduler = scheduler

    def handle_addprocess(self, args: List[str]) -> None:
        """Handle addprocess command to add a process to the scheduler"""
        if len(args) < 3:
            raise ValueError("addprocess: missing arguments\nUsage: addprocess <name> <duration> [priority]")

        name = args[1]
        if not name:
            raise ValueError("addprocess: process name cannot be empty")

        try:
            duration = float(args[2])
            if duration <= 0:
                raise ValueError("addprocess: duration must be positive")
        except ValueError as e:
            if "duration must be positive" in str(e):
                raise e
            raise ValueError(f"addprocess: invalid duration '{args[2]}': must be a number")

        priority = 0  # Default priority
        if len(args) > 3:
            try:
                priority = int(args[3])
            except ValueError:
                raise ValueError(f"addprocess: invalid priority '{args[3]}': must be an integer")

        try:
            pid = self.scheduler.add_process(name, duration, priority)
            if self.scheduler.config and self.scheduler.config.algorithm == SchedulingAlgorithm.PRIORITY:
                print(f"Added process '{name}' (PID: {pid}, Duration: {duration}s, Priority: {priority})")
            else:
                print(f"Added process '{name}' (PID: {pid}, Duration: {duration}s)")

        except ValueError as e:
            raise ValueError(str(e))

    def handle_scheduler(self, args: List[str]) -> None:
        """Handle scheduler command for all scheduler operations"""
        if len(args) < 2:
            raise ValueError("scheduler: missing subcommand\nUsage: scheduler <config|start|stop|status|metrics|clear|addprocess>")

        subcommand = args[1].lower()

        if subcommand == "config" or subcommand == "configure":
            self._handle_scheduler_config(args[2:])

        elif subcommand == "addprocess" or subcommand == "add":
            self._handle_scheduler_addprocess(args[2:])

        elif subcommand == "start":
            try:
                self.scheduler.start_scheduler()
            except ValueError as e:
                raise ValueError(f"scheduler start: {e}")

        elif subcommand == "stop":
            if self.scheduler.running:
                self.scheduler.stop_scheduler()
                print("Scheduler stopped")
            else:
                print("Scheduler is not running")

        elif subcommand == "clear":
            if self.scheduler.running:
                raise ValueError("scheduler clear: Cannot clear state while scheduler is running. Stop scheduler first.")

            self.scheduler.clear_scheduler_state()
            print("Scheduler state and metrics cleared")
            print("Use 'scheduler config <algorithm>' to configure the scheduler")

        elif subcommand == "status":
            status = self.scheduler.get_status()
            self._print_scheduler_status(status)

        elif subcommand == "metrics":
            metrics = self.scheduler.get_metrics()
            self._print_scheduler_metrics(metrics)

        else:
            raise ValueError(f"scheduler: unknown subcommand '{subcommand}'\nAvailable subcommands: config, addprocess, start, stop, status, metrics, clear")

    def _handle_scheduler_config(self, args: List[str]) -> None:
        """Handle scheduler config subcommand"""
        if len(args) < 1:
            raise ValueError("scheduler config: missing algorithm\nUsage: scheduler config <rr|priority> [time_quantum]")

        algorithm_str = args[0].lower()

        if algorithm_str == "rr" or algorithm_str == "round_robin":
            algorithm = SchedulingAlgorithm.ROUND_ROBIN
            time_quantum = 1.0  # Default time quantum

            if len(args) > 1:
                try:
                    time_quantum = float(args[1])
                    if time_quantum <= 0:
                        raise ValueError("scheduler config: time quantum must be positive")
                except ValueError as e:
                    if "time quantum must be positive" in str(e):
                        raise e
                    raise ValueError(f"scheduler config: invalid time quantum '{args[1]}': must be a number")

            self.scheduler.configure(algorithm, time_quantum)
            print(f"Configured Round-Robin scheduling with time quantum: {time_quantum}s")

        elif algorithm_str == "priority":
            algorithm = SchedulingAlgorithm.PRIORITY
            self.scheduler.configure(algorithm)
            print("Configured Priority-Based scheduling")

        else:
            raise ValueError(f"scheduler config: unknown algorithm '{algorithm_str}'\nSupported algorithms: rr (round_robin), priority")

    def _handle_scheduler_addprocess(self, args: List[str]) -> None:
        """Handle scheduler addprocess subcommand"""
        if len(args) < 2:
            raise ValueError("scheduler addprocess: missing arguments\nUsage: scheduler addprocess <name> <duration> [priority]")

        name = args[0]
        if not name:
            raise ValueError("scheduler addprocess: process name cannot be empty")

        try:
            duration = float(args[1])
            if duration <= 0:
                raise ValueError("scheduler addprocess: duration must be positive")
        except ValueError as e:
            if "duration must be positive" in str(e):
                raise e
            raise ValueError(f"scheduler addprocess: invalid duration '{args[1]}': must be a number")

        priority = 0  # Default priority
        if len(args) > 2:
            try:
                priority = int(args[2])
            except ValueError:
                raise ValueError(f"scheduler addprocess: invalid priority '{args[2]}': must be an integer")

        try:
            pid = self.scheduler.add_process(name, duration, priority)
            if self.scheduler.config and self.scheduler.config.algorithm == SchedulingAlgorithm.PRIORITY:
                print(f"Added process '{name}' (PID: {pid}, Duration: {duration}s, Priority: {priority})")
            else:
                print(f"Added process '{name}' (PID: {pid}, Duration: {duration}s)")

        except ValueError as e:
            raise ValueError(str(e))

    def _print_scheduler_status(self, status: dict) -> None:
        """Print scheduler status information"""
        print("=== Scheduler Status ===")

        if not status['algorithm']:
            print("Status: Not configured")
            print("Use 'scheduler config <rr|priority>' to configure the scheduler")
            return

        algorithm_name = "Round-Robin" if status['algorithm'] == 'round_robin' else "Priority-Based"
        print(f"Algorithm: {algorithm_name}")

        if status['algorithm'] == 'round_robin':
            print(f"Time Quantum: {status['time_quantum']}s")

        print(f"Status: {'Running' if status['running'] else 'Stopped'}")
        print(f"Completed Processes: {status['completed_processes']}")
        print(f"Total Context Switches: {status['total_context_switches']}")

        if status['current_process']:
            cp = status['current_process']
            if status['algorithm'] == 'priority':
                print(f"Current Process: PID {cp['pid']} ({cp['name']}) - Remaining: {cp['remaining_time']}s, Priority: {cp['priority']}")
            else:
                print(f"Current Process: PID {cp['pid']} ({cp['name']}) - Remaining: {cp['remaining_time']}s")
        else:
            print("Current Process: None")

        print(f"Ready Queue Size: {status['ready_queue_size']}")

        if status.get('ready_processes'):
            print("Ready Processes:")
            for proc in status['ready_processes']:
                if 'priority' in proc:
                    print(f"  PID {proc['pid']} ({proc['name']}) - Remaining: {proc['remaining_time']}s, Priority: {proc['priority']}")
                else:
                    print(f"  PID {proc['pid']} ({proc['name']}) - Remaining: {proc['remaining_time']}s")

        # Show helpful commands if no processes are queued
        if status['ready_queue_size'] == 0 and not status['current_process'] and not status['running']:
            print()
            print("Tip: Use 'addprocess <name> <duration> [priority]' to add processes")
            print("     Use 'scheduler start' to begin scheduling")

    def _print_scheduler_metrics(self, metrics: dict) -> None:
        """Print scheduler performance metrics"""
        print("=== Scheduler Performance Metrics ===")

        if 'message' in metrics:
            print(metrics['message'])
            if not metrics.get('completed_processes', 0):
                print()
                print("Tip: Add processes with 'scheduler addprocess' and run 'scheduler start'")
                print("     to generate performance metrics")
            return

        print(f"Completed Processes: {metrics['completed_processes']}")
        print(f"Total Context Switches: {metrics['total_context_switches']}")
        print()
        print("Average Times:")
        print(f"  Waiting Time: {metrics['average_waiting_time']}s")
        print(f"  Turnaround Time: {metrics['average_turnaround_time']}s")
        print(f"  Response Time: {metrics['average_response_time']}s")

        if metrics['process_details']:
            print()
            print("Individual Process Metrics:")
            print("PID | Name        | Waiting | Turnaround | Response")
            print("----|-------------|---------|------------|----------")
            for proc in metrics['process_details']:
                name = proc['name'][:11]  # Truncate long names
                print(f"{proc['pid']:3} | {name:<11} | {proc['waiting_time']:7.2f} | {proc['turnaround_time']:10.2f} | {proc['response_time']:8.2f}")

        print()
        print("Use 'scheduler clear' to reset metrics and start fresh")