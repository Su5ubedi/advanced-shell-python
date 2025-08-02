#!/usr/bin/env python3
"""
input_handler.py - Enhanced input handling with keyboard navigation
"""

import sys
import platform
from typing import Tuple

# Platform-specific imports with fallbacks
try:
    import tty
    import termios
    TTY_AVAILABLE = True
except ImportError:
    TTY_AVAILABLE = False

try:
    import msvcrt
    MSVCRT_AVAILABLE = True
except ImportError:
    MSVCRT_AVAILABLE = False


class InputHandler:
    """Handles enhanced input with keyboard navigation support"""

    def __init__(self):
        self.history = []
        self.history_index = 0
        self.max_history = 100
        self.is_windows = platform.system() == "Windows"
        self.enhanced_input_available = TTY_AVAILABLE or MSVCRT_AVAILABLE

    def get_char(self) -> str:
        """Get a single character from stdin"""
        if self.is_windows and MSVCRT_AVAILABLE:
            # Windows implementation using msvcrt
            while True:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()

                    # Handle special Windows keys
                    if ch == b'\x00' or ch == b'\xe0':  # Special key prefix
                        ch2 = msvcrt.getch()
                        # Arrow keys
                        if ch2 == b'H':  # Up arrow
                            return '\x1b[A'
                        elif ch2 == b'P':  # Down arrow
                            return '\x1b[B'
                        elif ch2 == b'M':  # Right arrow
                            return '\x1b[C'
                        elif ch2 == b'K':  # Left arrow
                            return '\x1b[D'
                        else:
                            continue  # Skip other special keys

                    # Convert bytes to string
                    if isinstance(ch, bytes):
                        try:
                            return ch.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                return ch.decode('cp1252')
                            except UnicodeDecodeError:
                                continue  # Skip invalid characters

                    return ch

        elif TTY_AVAILABLE:
            # Unix implementation using termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
        else:
            # Fallback for systems without enhanced input support
            try:
                return input()[0] if input() else '\r'
            except (EOFError, IndexError):
                return '\r'

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

    def clear_current_command(self, text: str, cursor_pos: int) -> Tuple[str, int]:
        """Clear the current command (Ctrl+C)"""
        self.clear_line()
        return "", 0

    def backspace(self, text: str, pos: int) -> Tuple[str, int]:
        """Handle backspace key"""
        if pos > 0:
            text = text[:pos-1] + text[pos:]
            pos -= 1

            sys.stdout.write('\b \b')
            if pos < len(text):
                sys.stdout.write(text[pos:])
                sys.stdout.write(' ')
                sys.stdout.write('\b' * (len(text) - pos + 1))
            sys.stdout.flush()

            return text, pos
        return text, pos

    def get_input(self, prompt: str = "") -> str:
        """Get input with keyboard navigation support"""
        if prompt:
            sys.stdout.write(prompt)
            sys.stdout.flush()

        # If enhanced input is not available, fall back to regular input
        if not self.enhanced_input_available:
            try:
                line = input()
                if line.strip():
                    self.add_to_history(line.strip())
                return line
            except EOFError:
                raise EOFError()
            except KeyboardInterrupt:
                print('^C')
                return ""

        text = ""
        cursor_pos = 0

        while True:
            try:
                ch = self.get_char()

                # Handle special keys
                if ch == '\x1b':  # ESC - might be arrow key sequence
                    try:
                        ch2 = self.get_char()
                        if ch2 == '[':
                            ch3 = self.get_char()
                            if ch3 == 'A':  # UP arrow
                                continue  # History navigation (future enhancement)
                            elif ch3 == 'B':  # DOWN arrow
                                continue  # History navigation (future enhancement)
                            elif ch3 == 'C':  # RIGHT arrow
                                cursor_pos = self.move_cursor_right(cursor_pos, text)
                                continue
                            elif ch3 == 'D':  # LEFT arrow
                                cursor_pos = self.move_cursor_left(cursor_pos)
                                continue
                    except:
                        # If we can't read the escape sequence, just ignore it
                        continue
                    continue

                # Handle arrow keys from Windows (already converted)
                elif ch == '\x1b[A':  # UP arrow
                    continue  # History navigation (future enhancement)
                elif ch == '\x1b[B':  # DOWN arrow
                    continue  # History navigation (future enhancement)
                elif ch == '\x1b[C':  # RIGHT arrow
                    cursor_pos = self.move_cursor_right(cursor_pos, text)
                    continue
                elif ch == '\x1b[D':  # LEFT arrow
                    cursor_pos = self.move_cursor_left(cursor_pos)
                    continue

                # Handle control characters
                elif ch == '\x03':  # Ctrl+C
                    text, cursor_pos = self.clear_current_command(text, cursor_pos)
                    print('^C')
                    continue
                elif ch == '\x04':  # Ctrl+D
                    print('^D')
                    raise EOFError()
                elif ch == '\x7f' or ch == '\x08':  # Backspace (Unix) or Backspace (Windows)
                    text, cursor_pos = self.backspace(text, cursor_pos)
                    continue
                elif ch == '\r' or ch == '\n':  # Enter
                    print()
                    if text.strip():
                        self.add_to_history(text.strip())
                    return text
                elif ch == '\t':  # Tab
                    continue  # Tab completion (future enhancement)
                elif ord(ch) < 32 or ord(ch) > 126:  # Non-printable characters
                    continue

                # Handle regular characters
                else:
                    text = self.insert_char(text, cursor_pos, ch)
                    sys.stdout.write(ch)
                    if cursor_pos < len(text) - 1:
                        sys.stdout.write(text[cursor_pos + 1:])
                        sys.stdout.write('\b' * (len(text) - cursor_pos - 1))
                    cursor_pos += 1
                    sys.stdout.flush()

            except (EOFError, KeyboardInterrupt) as e:
                if isinstance(e, EOFError):
                    print('\n^D')
                    raise
                else:
                    print('\n^C')
                    return ""
            except Exception:
                # Handle any other errors gracefully
                continue

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