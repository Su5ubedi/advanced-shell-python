#!/usr/bin/env python3
"""
command_parser.py - Command parsing and validation for Advanced Shell Simulation
"""

import shlex
from typing import Optional

from shell_types import ParsedCommand


class CommandParser:
    """Handles parsing of command line input"""

    BUILTIN_COMMANDS = {
        'cd', 'pwd', 'exit', 'echo', 'clear', 'ls', 'cat',
        'mkdir', 'rmdir', 'rm', 'touch', 'kill', 'jobs',
        'fg', 'bg', 'stop', 'help'
    }

    def parse(self, input_str: str) -> Optional[ParsedCommand]:
        """Parse a command line input string"""
        input_str = input_str.strip()
        if not input_str:
            return None

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
            pipes=[args]  # Single command for now
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
        dangerous_chars = '|;&<>(){}[]'
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