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
- **stop [job_id]** - Stop a running job (for testing bg command)
- **exit** - Exit shell
- **help** - Show available commands

### Process Management & Job Control
- **jobs** - List background jobs
- **fg [job_id]** - Bring background job to foreground
- **bg [job_id]** - Resume stopped job in background
- Signal handling and error management

## Project Structure

```
advanced-shell/
├── main.py              # Entry point with command-line options
├── shell.py             # Main shell class and command loop
├── command_handler.py   # Built-in command implementations
├── command_parser.py    # Command parsing and validation
├── job_manager.py       # Job control operations
├── types.py             # Data types and enums
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

# Process and job management
kill 1234          # Kill process by PID
sleep 60 &         # Start background job
stop 1             # Stop job 1 (testing feature)
bg 1               # Resume job 1 in background
fg 1               # Bring job 1 to foreground

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
- **stop [job_id]** - Stop a running job (for testing bg command)
- **exit** - Exit shell
- **help** - Show available commands

### Process Management & Job Control
- **jobs** - List background jobs
- **fg [job_id]** - Bring background job to foreground
- **bg [job_id]** - Resume stopped job in background
- Signal handling and error management

## Project Structure

```
advanced-shell/
├── main.py              # Entry point with command-line options
├── shell.py             # Main shell class and command loop
├── command_handler.py   # Built-in command implementations
├── command_parser.py    # Command parsing and validation
├── job_manager.py       # Job control operations
├── types.py             # Data types and enums
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

# Process and job management
kill 1234          # Kill process by PID
sleep 60 &         # Start background job
stop 1             # Stop job 1 (testing feature)
bg 1               # Resume job 1 in background
fg 1               # Bring job 1 to foreground

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

# Test job control scenarios (Deliverable 1 - preparation for future)
jobs               # List jobs (shows "No active jobs")
fg                 # Missing job ID error
fg abc             # Invalid job ID error
fg 999             # Job not found error
bg 1               # Job not found error
stop               # Missing job ID error
stop abc           # Invalid job ID error
stop 999           # Job not found error

# Test process management
kill               # Missing PID error
kill abc           # Invalid PID error
kill -1            # Invalid PID error
kill 1             # Cannot kill init process error
kill 12345         # Attempt to kill process (if PID exists)

# Test error handling
invalid_command    # Command not found (only built-in commands supported)
cp file1 file2     # Command not found error
echo               # Works (empty output)
cd                 # Goes to home directory
pwd                # Shows current directory

# Test comprehensive file operations workflow
mkdir test_project
cd test_project
touch README.md
touch main.py
ls -la             # Shows created files
echo "Hello World" > temp.txt  # Note: redirection not implemented yet
cat README.md      # Shows file content (empty)
rm main.py
rmdir ../test_project  # Remove directory (should fail - not empty)
cd ..
rm -rf test_project    # Remove directory recursively
pwd                    # Verify current location

# Test job control workflow
sleep 60 &         # Start background job
jobs               # Shows running job
stop 1             # Stop the job
jobs               # Shows stopped job
bg 1               # Resume job in background
fg 1               # Bring to foreground (Ctrl+C to interrupt)

# Note: Background job execution (&) works with external commands
# The 'stop' command is a testing feature to demonstrate bg/fg functionality
# In future deliverables, job control will be enhanced with proper scheduling
```