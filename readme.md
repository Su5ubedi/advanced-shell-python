# Advanced Shell Simulation - Deliverable 1

A custom shell implementation in Python that simulates Unix-like Operating System environment with process management, job control, and built-in commands.

## Features Implemented (Deliverable 1)

### Built-in Commands
- **cd [directory]** - Change directory (supports ~, relative paths)
- **pwd** - Print working directory
- **echo [text]** - Print text (supports \n, \t escape sequences)
- **clear** - Clear screen
- **ls [options] [dir]** - List files (-a for hidden, -l for long format)
- **cat [files...]** - Display file contents
- **mkdir [options] [dirs...]** - Create directories (-p for parents)
- **rmdir [dirs...]** - Remove empty directories
- **rm [options] [files...]** - Remove files (-r recursive, -f force)
- **touch [files...]** - Create empty files or update timestamps
- **kill [pids...]** - Kill processes by PID
- **exit** - Exit shell
- **help** - Show available commands

### Process Management & Job Control
- **External command execution** - Run any system command
- **Foreground execution** - Commands run in foreground by default
- **Background execution** - Add `&` to run commands in background
- **jobs** - List all background jobs with status
- **fg [job_id]** - Bring background job to foreground
- **bg [job_id]** - Resume stopped job in background
- **stop [job_id]** - Stop a running job (equivalent to Ctrl+Z)
- **Signal handling** - Proper handling of SIGINT, SIGCHLD
- **Process groups** - Support for process group management
- **Error management** - Comprehensive error handling

### Keyboard Navigation
- **Arrow Keys** - Navigate cursor left and right in command line
- **Ctrl+C** - Clear current command line (when typing)
- **Ctrl+C** - Interrupt current foreground process (when running)
- **Backspace** - Delete character to the left of cursor
- **Ctrl+D** - Exit shell (EOF)

## Project Structure

```
advanced-shell/
├── main.py              # Entry point with command-line options
├── shell.py             # Main shell class and command loop
├── command_handler.py   # Built-in command implementations
├── command_parser.py    # Command parsing and validation
├── job_manager.py       # Job control operations
├── input_handler.py     # Enhanced input with keyboard navigation
├── shell_types.py       # Data types and enums
└── README.md           # Documentation
```

## How to Run

### Prerequisites
- Python 3.8 or higher

### Running the Shell
```bash
# Start the shell
python3 main.py

# Command-line options
python3 main.py --help     # Show help
python3 main.py --version  # Show version info
python3 main.py --debug    # Enable debug mode
```

### Example Usage
```bash
# Navigation and file operations
ls -la
pwd
cd /tmp
mkdir -p test/nested/path
touch test_file.txt
cat test_file.txt
rm -rf test

# Directory operations
mkdir my_directory
rmdir my_directory

# Text operations
echo "Hello World"
echo "Line 1\nLine 2"

# Process management
kill 1234

# Shell operations
help
exit
```

## Future Deliverables

This implementation provides the foundation for upcoming features:

### Deliverable 2: Process Scheduling
- Round-Robin Scheduling with configurable time slices
- Priority-Based Scheduling with preemption
- Performance metrics (waiting time, turnaround time, response time)

### Deliverable 3: Memory Management & Synchronization
- Paging system with FIFO and LRU page replacement
- Process synchronization with mutexes/semaphores
- Classical synchronization problems (Producer-Consumer, Dining Philosophers)

### Deliverable 4: Integration & Security
- Command piping support
- User authentication system
- File permissions and access control
- Complete system integration

## Architecture

### Modular Design
- **Shell**: Main command loop and user interaction
- **CommandParser**: Input tokenization and validation
- **CommandHandler**: Built-in command implementations
- **JobManager**: Background process and job control
- **Types**: Data structures for jobs and commands

### Error Handling
Comprehensive error handling for:
- Invalid commands and arguments
- File/directory permissions
- Process management errors
- Input validation errors

## Testing

Test the shell with various scenarios:
```bash
# Test built-in commands
ls -xyz            # Invalid option error
mkdir              # Missing argument error
cd /nonexistent    # No such directory error

# Test file operations
touch test.txt
cat test.txt
rm test.txt

# Test directory operations
mkdir test_dir
ls -la
rmdir test_dir

# Test job control with background processes
# Note: Use system commands that support background execution
sleep 30 &         # Start background job - shows [1] PID
jobs               # List active jobs - shows job [1] Running
fg 1               # Bring job 1 to foreground (Ctrl+C to interrupt)
sleep 60 &         # Start another background job
ping localhost &   # Start third background job
jobs               # Shows multiple running jobs
bg 2               # Resume job 2 in background (if stopped)
fg 3               # Bring job 3 to foreground

# Test job control error scenarios
jobs               # List jobs (may show "No active jobs" if none running)
fg                 # Missing job ID error
fg abc             # Invalid job ID error
fg 999             # Job not found error
bg 1               # Job not found error (if no job 1)

# Test process management
kill               # Missing PID error
kill abc           # Invalid PID error
kill -1            # Invalid PID error
kill 1             # Cannot kill init process error

# Test error handling
invalid_command    # Command not found error
echo               # Works (empty output)
cd                 # Goes to home directory
pwd                # Shows current directory

# Complete job workflow example
sleep 45 &         # [1] 12345 (job 1 started with PID 12345)
jobs               # [1] Running sleep 45 (PID: 12345, Duration: 2s)
fg 1               # Bringing job [1] to foreground: sleep 45
                   # (Press Ctrl+C to interrupt)
jobs               # No active jobs (job completed)
```