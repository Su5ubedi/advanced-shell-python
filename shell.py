#!/usr/bin/env python3
"""
shell.py - Main shell class for Advanced Shell Simulation
Updated for Deliverable 3: Memory Management and Process Synchronization
"""

import os
import signal
import subprocess
import time

from job_manager import JobManager
from command_handler import CommandHandler
from command_parser import CommandParser
from shell_types import ParsedCommand
from input_handler import InputHandler
from auth_system import Permission


class Shell:
    """Main shell implementation"""

    def __init__(self):
        self.job_manager = JobManager()
        self.command_handler = CommandHandler(self.job_manager)
        self.parser = CommandParser()
        self.input_handler = InputHandler()
        self.running = True
        self.prompt = "[shell]$ "
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def sigint_handler(signum, frame):
            print("\nReceived interrupt signal. Use 'exit' to quit the shell.")
            # Don't exit immediately, let user decide

        signal.signal(signal.SIGINT, sigint_handler)

    def run(self):
        """Start the main shell loop"""
        self.print_welcome()

        # Require login before starting shell
        if not self.command_handler.auth_system.is_authenticated():
            self.handle_initial_login()

        while self.running:
            try:
                self.display_prompt()

                try:
                    # Use input handler if available, otherwise fallback to input()
                    if self.input_handler:
                        user_input = self.input_handler.get_input(self.prompt).strip()
                    else:
                        user_input = input().strip()
                except EOFError:
                    # Handle Ctrl+D
                    print("\nGoodbye!")
                    break
                except KeyboardInterrupt:
                    # Handle Ctrl+C
                    print()
                    continue

                # Handle empty input
                if not user_input:
                    continue

                # Process the input and handle errors gracefully
                try:
                    self.process_input(user_input)
                except Exception as e:
                    print(f"\033[31mError:\033[0m {e}")

                # Clean up completed jobs after each command
                self.job_manager.cleanup_completed_jobs()

            except Exception as e:
                print(f"\033[31mShell Error:\033[0m {e}")

        self.shutdown()

    def handle_initial_login(self):
        """Handle initial login before starting shell"""
        print("Welcome to Advanced Shell Simulation - Deliverable 3")
        print("You must log in to continue.")
        print()
        print("Available users:")
        print("  admin/admin123    - Administrator (full access)")
        print("  user/user123      - Standard user (limited access)")
        print("  guest/guest123    - Guest user (very limited access)")
        print()

        while not self.command_handler.auth_system.is_authenticated():
            try:
                username = input("Username: ").strip()
                if not username:
                    continue

                password = input("Password: ").strip()
                if not password:
                    continue

                if self.command_handler.auth_system.authenticate(username, password):
                    user = self.command_handler.auth_system.get_current_user()
                    print(f"\nWelcome, {user.username}! Role: {user.role.value}")
                    break
                else:
                    print("Login failed: Invalid username or password")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                sys.exit(0)
            except EOFError:
                print("\nGoodbye!")
                sys.exit(0)

    def process_input(self, input_str: str) -> None:
        """Process a single line of input"""
        # Parse the command
        try:
            parsed = self.parser.parse(input_str)
        except ValueError as e:
            raise ValueError(f"Parse error: {e}")

        if not parsed:
            return  # Empty command

        # Validate the command
        try:
            self.parser.validate_command(parsed)
        except ValueError as e:
            raise ValueError(f"Validation error: {e}")

        # Handle piped commands
        if parsed.has_pipes:
            try:
                result = self.command_handler.handle_pipe_command(parsed)
                if result:
                    print(result, end='')
            except Exception as e:
                raise ValueError(f"Pipe error: {e}")
            return

        # Check if it's a built-in command
        if self.parser.is_builtin_command(parsed.command):
            try:
                self.command_handler.handle_command(parsed)
            except ValueError as e:
                raise ValueError(str(e))
            except Exception as e:
                raise ValueError(f"{parsed.command}: {e}")
        else:
            # Try to execute as external command
            self.execute_external_command(parsed)

    def execute_external_command(self, parsed: ParsedCommand) -> None:
        """Execute external command with permission checking"""
        # Check if external command exists
        import shutil
        if not shutil.which(parsed.command):
            raise ValueError(f"{parsed.command}: command not found")

        # Check permissions for command execution
        if not self.command_handler.auth_system.is_authenticated():
            raise ValueError("Authentication required to execute commands")

        # Check if user has execute permission for the command
        command_path = shutil.which(parsed.command)
        if command_path and not self.command_handler.check_file_permission(command_path, Permission.EXECUTE):
            raise PermissionError(f"{parsed.command}: permission denied")

        try:
            if parsed.background:
                # Background execution
                process = subprocess.Popen(
                    parsed.args,
                    cwd=os.getcwd(),
                    preexec_fn=os.setsid  # Create new process group
                )

                job = self.job_manager.add_job(
                    command=' '.join(parsed.args),
                    args=parsed.args,
                    process=process,
                    background=True
                )

                print(f"[{job.id}] {process.pid}")

                # Deliverable 3: NEW - Integrate with memory management for external commands
                try:
                    memory_manager = self.command_handler.get_memory_manager()
                    if memory_manager:
                        # Estimate memory needs for external command
                        pages_needed = self._estimate_memory_needs(parsed.command)
                        memory_pid = memory_manager.create_process(f"ExtCmd-{parsed.command}", pages_needed)
                except:
                    pass  # Don't fail if memory management unavailable

            else:
                # Foreground execution
                process = subprocess.Popen(
                    parsed.args,
                    cwd=os.getcwd()
                )

                try:
                    process.wait()
                except KeyboardInterrupt:
                    # Forward Ctrl+C to child process
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    print("\n^C")

        except FileNotFoundError:
            raise ValueError(f"{parsed.command}: command not found")
        except PermissionError:
            raise ValueError(f"{parsed.command}: permission denied")
        except Exception as e:
            raise ValueError(f"Error executing {parsed.command}: {e}")

    def _estimate_memory_needs(self, command: str) -> int:
        """Estimate memory needs for external commands """
        # Simple heuristic for memory estimation
        memory_estimates = {
            'ls': 2, 'grep': 3, 'sort': 5, 'cat': 2, 'find': 4,
            'python': 8, 'gcc': 10, 'make': 6, 'vim': 4, 'nano': 2
        }
        return memory_estimates.get(command, 4)  # Default to 4 pages

    def display_prompt(self) -> None:
        """Show the shell prompt with user information"""
        try:
            pwd = os.getcwd()
            dir_name = os.path.basename(pwd)
            if dir_name == "":
                dir_name = pwd
        except Exception:
            dir_name = "unknown"

        # Get current time for enhanced prompt
        current_time = time.strftime("%H:%M:%S")

        # Add user information to prompt
        if self.command_handler.auth_system.is_authenticated():
            user = self.command_handler.auth_system.get_current_user()
            self.prompt = f"[{user.username}:{user.role.value}:{dir_name} {current_time}]$ "
        else:
            self.prompt = f"[shell:{dir_name} {current_time}]$ "

    def print_welcome(self) -> None:
        """Print the welcome message"""
        print("==========================================")
        print("  Advanced Shell Simulation - Deliverable 3")
        print("==========================================")
        print()
        print("Features implemented:")
        print("✓ Built-in commands (cd, pwd, ls, cat, etc.)")
        print("✓ External command execution")
        print("✓ Process management (foreground/background)")
        print("✓ Job control (jobs, fg, bg, stop)")
        print("✓ Round-Robin scheduling with configurable time slices")
        print("✓ Priority-based scheduling with preemption")
        print("✓ Process simulation using timers")
        print("✓ Keyboard navigation (arrow keys, Ctrl+C to clear)")
        print("✓ Signal handling")
        print("✓ Error handling")
        print("✓ Process scheduling algorithms")
        print("  • Round-Robin Scheduling")
        print("  • Priority-Based Scheduling")
        print("✓ Performance metrics and monitoring")
        print("✓ Memory management with paging ")
        print("  • FIFO and LRU page replacement algorithms")
        print("  • Page fault handling and tracking")
        print("  • Memory overflow simulation")
        print("✓ Process synchronization ")
        print("  • Mutexes and semaphores")
        print("  • Producer-Consumer problem")
        print("  • Dining Philosophers problem")
        print("  • Race condition prevention")
        print("  • Deadlock avoidance")
        print("✓ Command piping and redirection ")
        print("✓ User authentication system ")
        print("✓ File permissions and access control ")
        print("✓ READ-ONLY access for standard users (UPDATED)")
        print()
        print("Type 'help' for available commands")
        print("Type 'exit' to quit")
        print()
        print("Quick Start - Authentication:")
        print("  login admin admin123    # Login as administrator (full access)")
        print("  login user user123      # Login as standard user (read-only)")
        print("  login guest guest123    # Login as guest (very limited)")
        print("  whoami                  # Show current user")
        print("  logout                  # Logout current user")
        print()
        print("Quick Start - File Permissions:")
        print("  permissions file.txt    # Check file permissions")
        print("  ls -l                   # List with permission details")
        print()
        print("Quick Start - Piping:")
        print("  ls | grep txt           # List files containing 'txt'")
        print("  cat file.txt | sort     # Display sorted file contents")
        print()
        print("Quick Start - Process Scheduling:")
        print("  scheduler config rr 2         # Configure Round-Robin")
        print("  scheduler addprocess task1 5  # Add 5-second process")
        print("  scheduler start               # Start scheduling")
        print("  scheduler status              # Monitor execution")
        print()
        print("✓ Quick Start - Memory Management :")
        print("  memory create webapp 8        # Create process needing 8 pages")
        print("  memory alloc 1 0              # Allocate page 0 for process 1")
        print("  memory algorithm lru          # Switch to LRU replacement")
        print("  memory status                 # Show memory statistics")
        print()
        print("✓ Quick Start - Synchronization :")
        print("  sync prodcons start 2 3       # Start Producer-Consumer")
        print("  sync philosophers start 5     # Start Dining Philosophers")
        print("  sync status                   # Show sync statistics")
        print("IMPORTANT: Standard and Guest users have READ-ONLY access")
        print("to most directories. Only Admin can create/modify files.")
        print()

    def shutdown(self) -> None:
        """Perform cleanup before exiting"""
        print("\nShutting down shell...")

        # Deliverable 3: NEW - Stop synchronization problems
        try:
            memory_sync_commands = getattr(self.command_handler, 'memory_sync_commands', None)
            if memory_sync_commands:
                if memory_sync_commands.producer_consumer:
                    print("Stopping Producer-Consumer...")
                    memory_sync_commands.producer_consumer.stop()

                if memory_sync_commands.dining_philosophers:
                    print("Stopping Dining Philosophers...")
                    memory_sync_commands.dining_philosophers.stop()
        except:
            pass
        # Cleanup pipe handler
        if hasattr(self.command_handler, 'pipe_handler'):
            self.command_handler.pipe_handler.cleanup()

        # Stop scheduler if available
        if hasattr(self.command_handler, 'process_scheduler') and self.command_handler.process_scheduler:
            print("Stopping process scheduler...")
            self.command_handler.process_scheduler.stop_scheduler()

        # Get all active jobs
        jobs = self.job_manager.get_all_jobs()
        if jobs:
            print(f"Terminating {len(jobs)} active job(s)...")

            for job in jobs:
                if job.status.value != "Done":
                    print(f"Killing job [{job.id}]: {job.command}")
                    self.job_manager.kill_job(job.id)

            # Give processes time to terminate
            time.sleep(0.1)

        print("Goodbye!")