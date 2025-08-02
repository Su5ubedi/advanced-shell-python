#!/usr/bin/env python3
"""
input_handler.py - Enhanced input handling with keyboard navigation
"""

import sys
import tty
import termios
import os
from typing import Optional


class InputHandler:
    """Handles enhanced input with keyboard navigation support"""
    
    def __init__(self):
        self.history = []
        self.history_index = 0
        self.max_history = 100
        
    def get_char(self) -> str:
        """Get a single character from stdin"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    

    
    def clear_line(self):
        """Clear the current line"""
        sys.stdout.write('\r\033[K')
        sys.stdout.flush()
    
    def move_cursor_left(self, current_pos: int) -> int:
        """Move cursor left if possible"""
        if current_pos > 0:
            sys.stdout.write('\b')
            sys.stdout.flush()
            return current_pos - 1
        return current_pos
    
    def move_cursor_right(self, current_pos: int, text: str) -> int:
        """Move cursor right if possible"""
        if current_pos < len(text):
            sys.stdout.write(text[current_pos])
            sys.stdout.flush()
            return current_pos + 1
        return current_pos
    
    def clear_current_command(self, text: str, cursor_pos: int) -> tuple[str, int]:
        """Clear the current command (Ctrl+C)"""
        # Clear the entire line
        self.clear_line()
        return "", 0
    
    def backspace(self, text: str, pos: int) -> tuple[str, int]:
        """Handle backspace key"""
        if pos > 0:
            # Remove the character
            text = text[:pos-1] + text[pos:]
            pos -= 1
            
            # Clear the character and redraw the rest of the line
            sys.stdout.write('\b \b')
            if pos < len(text):
                sys.stdout.write(text[pos:])
                sys.stdout.write(' ')
                # Move cursor back to the correct position
                sys.stdout.write('\b' * (len(text) - pos + 1))
            sys.stdout.flush()
            
            return text, pos
        return text, pos
    
    def get_input(self, prompt: str = "") -> str:
        """Get input with keyboard navigation support"""
        if prompt:
            sys.stdout.write(prompt)
            sys.stdout.flush()
        
        text = ""
        cursor_pos = 0
        
        while True:
            ch = self.get_char()
            
            # Handle special keys
            if ch == '\x1b':  # ESC - might be arrow key
                ch2 = self.get_char()
                if ch2 == '[':
                    ch3 = self.get_char()
                    if ch3 == 'A':  # UP arrow
                        # History navigation (future enhancement)
                        continue
                    elif ch3 == 'B':  # DOWN arrow
                        # History navigation (future enhancement)
                        continue
                    elif ch3 == 'C':  # RIGHT arrow
                        cursor_pos = self.move_cursor_right(cursor_pos, text)
                        continue
                    elif ch3 == 'D':  # LEFT arrow
                        cursor_pos = self.move_cursor_left(cursor_pos)
                        continue
                # If not an arrow key, ignore the ESC sequence
                continue
            
            # Handle control characters
            elif ch == '\x03':  # Ctrl+C - clear current command
                text, cursor_pos = self.clear_current_command(text, cursor_pos)
                print('^C')  # Show Ctrl+C was pressed
                continue
            elif ch == '\x04':  # Ctrl+D
                print('^D')
                raise EOFError()
            elif ch == '\x7f':  # Backspace
                text, cursor_pos = self.backspace(text, cursor_pos)
                continue
            elif ch == '\r':  # Enter
                print()
                if text.strip():
                    self.add_to_history(text.strip())
                return text
            elif ch == '\t':  # Tab
                # Tab completion (future enhancement)
                continue
            elif ch < ' ' or ch > '~':  # Non-printable characters
                continue
            
            # Handle regular characters
            else:
                text = self.insert_char(text, cursor_pos, ch)
                # Display the character and any text that follows
                sys.stdout.write(ch)
                if cursor_pos < len(text) - 1:
                    sys.stdout.write(text[cursor_pos + 1:])
                    # Move cursor back to the correct position
                    sys.stdout.write('\b' * (len(text) - cursor_pos - 1))
                cursor_pos += 1
                sys.stdout.flush()
    
    def insert_char(self, text: str, pos: int, char: str) -> str:
        """Insert a character at the specified position"""
        return text[:pos] + char + text[pos:]
    
    def add_to_history(self, command: str):
        """Add command to history"""
        if command and (not self.history or command != self.history[-1]):
            self.history.append(command)
            if len(self.history) > self.max_history:
                self.history.pop(0)
        self.history_index = len(self.history) 