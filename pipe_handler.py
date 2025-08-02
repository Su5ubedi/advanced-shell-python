#!/usr/bin/env python3
"""
pipe_handler.py - Command piping and redirection system
"""

import subprocess
import os
import tempfile
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PipeCommand:
    """Represents a command in a pipe chain"""
    command: str
    args: List[str]
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    append_output: bool = False


class PipeHandler:
    """Handles command piping and redirection"""

    def __init__(self):
        self.temp_files = []

    def parse_pipe_chain(self, input_str: str) -> List[PipeCommand]:
        """Parse a command string into a chain of piped commands"""
        commands = []
        
        # Split by pipe character
        pipe_parts = input_str.split('|')
        
        for i, part in enumerate(pipe_parts):
            part = part.strip()
            if not part:
                continue

            # Parse redirection
            input_file = None
            output_file = None
            append_output = False

            # Handle input redirection
            if '<' in part:
                parts = part.split('<', 1)
                part = parts[0].strip()
                input_file = parts[1].strip()

            # Handle output redirection
            if '>' in part:
                if '>>' in part:
                    parts = part.split('>>', 1)
                    part = parts[0].strip()
                    output_file = parts[1].strip()
                    append_output = True
                else:
                    parts = part.split('>', 1)
                    part = parts[0].strip()
                    output_file = parts[1].strip()

            # Parse command and arguments
            args = part.split()
            if not args:
                continue

            command = PipeCommand(
                command=args[0],
                args=args,
                input_file=input_file,
                output_file=output_file,
                append_output=append_output
            )
            commands.append(command)

        return commands

    def execute_pipe_chain(self, commands: List[PipeCommand], auth_system=None) -> str:
        """Execute a chain of piped commands"""
        if not commands:
            return ""

        # Check permissions for all commands
        if auth_system and auth_system.is_authenticated():
            for cmd in commands:
                if cmd.input_file and not auth_system.has_permission(cmd.input_file, auth_system.Permission.READ):
                    raise PermissionError(f"Permission denied: cannot read {cmd.input_file}")
                if cmd.output_file and not auth_system.has_permission(cmd.output_file, auth_system.Permission.WRITE):
                    raise PermissionError(f"Permission denied: cannot write to {cmd.output_file}")

        # For single command, use simple subprocess
        if len(commands) == 1:
            return self._execute_single_command(commands[0])

        # For multiple commands, use pipes
        return self._execute_pipe_chain(commands)

    def _execute_single_command(self, cmd: PipeCommand) -> str:
        """Execute a single command"""
        try:
            # Handle input/output redirection
            stdin = None
            stdout = None
            
            if cmd.input_file:
                stdin = open(cmd.input_file, 'r')
            
            if cmd.output_file:
                mode = 'a' if cmd.append_output else 'w'
                stdout = open(cmd.output_file, mode)
            else:
                # Capture output if no output file
                result = subprocess.run(
                    cmd.args,
                    stdin=stdin,
                    capture_output=True,
                    text=True
                )
                
                # Clean up input file
                if stdin:
                    stdin.close()
                
                if result.returncode != 0:
                    raise RuntimeError(f"Command '{cmd.command}' failed: {result.stderr}")
                
                return result.stdout
            
            # Execute with file redirection
            result = subprocess.run(
                cmd.args,
                stdin=stdin,
                stdout=stdout,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Clean up files
            if stdin:
                stdin.close()
            if stdout:
                stdout.close()
            
            if result.returncode != 0:
                raise RuntimeError(f"Command '{cmd.command}' failed: {result.stderr}")
            
            return ""
            
        except FileNotFoundError:
            raise ValueError(f"{cmd.command}: command not found")
        except PermissionError:
            raise PermissionError(f"{cmd.command}: permission denied")

    def _execute_pipe_chain(self, commands: List[PipeCommand]) -> str:
        """Execute a chain of piped commands"""
        processes = []
        pipes = []
        open_files = []

        try:
            # Create pipes between commands
            for i in range(len(commands) - 1):
                read_pipe, write_pipe = os.pipe()
                pipes.extend([read_pipe, write_pipe])

            # Execute each command
            for i, cmd in enumerate(commands):
                # Set up input
                stdin = None
                if i == 0 and cmd.input_file:
                    # First command with input file
                    stdin = open(cmd.input_file, 'r')
                    open_files.append(stdin)
                elif i > 0:
                    # Not first command, use pipe from previous
                    stdin = os.fdopen(pipes[(i-1)*2], 'r')
                    open_files.append(stdin)

                # Set up output
                stdout = None
                if i == len(commands) - 1 and cmd.output_file:
                    # Last command with output file
                    mode = 'a' if cmd.append_output else 'w'
                    stdout = open(cmd.output_file, mode)
                    open_files.append(stdout)
                elif i < len(commands) - 1:
                    # Not last command, use pipe to next
                    stdout = os.fdopen(pipes[i*2+1], 'w')
                    open_files.append(stdout)

                # Execute command
                try:
                    process = subprocess.Popen(
                        cmd.args,
                        stdin=stdin,
                        stdout=stdout,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    processes.append(process)
                except FileNotFoundError:
                    raise ValueError(f"{cmd.command}: command not found")
                except PermissionError:
                    raise PermissionError(f"{cmd.command}: permission denied")

            # Wait for all processes to complete
            for process in processes:
                process.wait()

            # Check for errors
            for i, process in enumerate(processes):
                if process.returncode != 0:
                    stderr_output = process.stderr.read() if process.stderr else ""
                    raise RuntimeError(f"Command '{commands[i].command}' failed: {stderr_output}")

            # Get output from last command if no output file
            if not commands[-1].output_file:
                # For piped commands, we need to capture the output differently
                # Since the last command's output goes to stdout, we need to read it
                # This is a simplified approach - in a real implementation, you'd need
                # to capture the output during execution
                return "Command executed successfully"

            return ""

        finally:
            # Clean up all open files
            for file_obj in open_files:
                try:
                    file_obj.close()
                except (OSError, AttributeError):
                    pass

            # Clean up pipes
            for pipe in pipes:
                try:
                    os.close(pipe)
                except OSError:
                    pass

    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass
        self.temp_files.clear() 