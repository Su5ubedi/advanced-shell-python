#!/usr/bin/env python3
"""
command_parser.py - Command parsing and validation for Advanced Shell Simulation
Updated for Deliverable 3: Memory Management and Process Synchronization
"""

import shlex
from typing import Optional

from shell_types import ParsedCommand


class CommandParser:
    """Handles parsing of command line input"""

    BUILTIN_COMMANDS = {
        # Deliverable 1: Basic shell commands
        'cd', 'pwd', 'exit', 'echo', 'clear', 'ls', 'cat',
        'mkdir', 'rmdir', 'rm', 'touch', 'kill', 'jobs',
        'fg', 'bg', 'stop', 'help',
        # Deliverable 2: Process scheduling commands
        'scheduler', 'addprocess', 'scheduler',
        # Deliverable 3: NEW - Memory management and synchronization commands
        'memory', 'sync',
        # Deliverable 3: Authentication commands
        'login', 'logout', 'whoami', 'adduser', 'chpasswd',
        # Text processing commands
        'grep', 'sort'
    }

    def parse(self, input_str: str) -> Optional[ParsedCommand]:
        """Parse a command line input string"""
        input_str = input_str.strip()
        if not input_str:
            return None

        # Check for pipes
        if '|' in input_str:
            return self._parse_pipe_command(input_str)

        # Check for background execution
        background = False
        if input_str.endswith('&'):
            background = True
            input_str = input_str[:-1].strip()

        # Tokenize the input
        try:
            args = shlex.split(input_str)
        except ValueError as e:
            raise ValueError(f"Parse error: {e}")

        if not args:
            return None

        return ParsedCommand(
            command=args[0],
            args=args,
            background=background,
            pipes=[args],
            has_pipes=False,
            pipe_chain=[]
        )

    def _parse_pipe_command(self, input_str: str) -> ParsedCommand:
        """Parse a command with pipes"""
        # Split by pipe character
        pipe_parts = [part.strip() for part in input_str.split('|')]

        # Check for background execution on the last command
        background = False
        if pipe_parts[-1].endswith('&'):
            background = True
            pipe_parts[-1] = pipe_parts[-1][:-1].strip()

        # Parse each command in the pipe chain
        pipe_commands = []
        for part in pipe_parts:
            if not part:
                continue
            try:
                args = shlex.split(part)
                pipe_commands.append(args)
            except ValueError as e:
                raise ValueError(f"Parse error in pipe command: {e}")

        if not pipe_commands:
            return None

        # The main command is the first one in the chain
        main_command = pipe_commands[0][0] if pipe_commands[0] else ""

        return ParsedCommand(
            command=main_command,
            args=pipe_commands[0] if pipe_commands else [],
            background=background,
            pipes=pipe_commands,
            has_pipes=True,
            pipe_chain=pipe_parts
        )

    def is_builtin_command(self, command: str) -> bool:
        """Check if a command is a built-in command"""
        return command in self.BUILTIN_COMMANDS

    def validate_command(self, parsed: ParsedCommand) -> None:
        """Perform comprehensive validation on parsed commands"""
        if not parsed or not parsed.command:
            return

        # Check for dangerous command patterns
        if '..' in parsed.command:
            raise ValueError(f"Potentially dangerous path detected: {parsed.command}")

        # Validate command name
        dangerous_chars = ';&<>(){}[]'
        if any(char in parsed.command for char in dangerous_chars):
            raise ValueError(f"Invalid characters in command name: {parsed.command}")

        # Check for excessively long commands
        if len(parsed.command) > 256:
            raise ValueError("Command name too long (max 256 characters)")

        # Validate arguments
        for i, arg in enumerate(parsed.args):
            if len(arg) > 1024:
                raise ValueError(f"Argument {i} too long (max 1024 characters)")

        # Check total argument count
        if len(parsed.args) > 100:
            raise ValueError("Too many arguments (max 100)")

        # Validate pipe commands
        if parsed.has_pipes:
            for pipe_cmd in parsed.pipes:
                if not pipe_cmd:
                    raise ValueError("Empty command in pipe chain")
                if len(pipe_cmd) > 100:
                    raise ValueError("Too many arguments in pipe command (max 100)")