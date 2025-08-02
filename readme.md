# Advanced Shell Simulation - Deliverable 2

A custom shell implementation in Python that simulates Unix-like Operating System environment with process management, job control, and advanced scheduling algorithms.

## Features Implemented (Deliverable 2)

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

### NEW: Process Scheduling (Deliverable 2)
- **schedule <command> [priority] [time]** - Add job to scheduler
- **scheduler [round_robin|priority] [time_slice]** - Configure scheduler
- **addjob <command> [priority] [time] [background]** - Add job with parameters

#### Scheduling Algorithms
1. **Round-Robin Scheduling**
   - Each process gets a configurable time slice
   - After time slice expires, process moves to end of queue
   - Time slice is configurable (default: 2.0 seconds)
   - Processes complete when their total time is reached

2. **Priority-Based Scheduling**
   - Processes run in priority order (1=highest, 10=lowest)
   - Same priority processes use First-Come-First-Served (FCFS)
   - Preemption: Higher priority jobs interrupt lower priority ones
   - Processes run to completion

#### Process Simulation
- Uses `time.sleep()` to simulate process execution
- Configurable execution time for each process
- Real-time process switching and status updates
- Thread-safe scheduling operations

## Project Structure

```
advanced-shell-python/
├── main.py              # Entry point with command-line options
├── shell.py             # Main shell class and command loop
├── command_handler.py   # Built-in command implementations
├── command_parser.py    # Command parsing and validation
├── job_manager.py       # Job control operations (enhanced with scheduling)
├── scheduler.py         # NEW: Process scheduling algorithms
├── shell_types.py       # Data types and enums (enhanced)
├── test_scheduling.py   # NEW: Test script for scheduling features
└── readme.md           # Documentation
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

### Testing Scheduling Features
```bash
# Run the scheduling test suite
python3 test_scheduling.py
```

## Example Usage

### Basic Shell Operations
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

### NEW: Scheduling Examples

#### Round-Robin Scheduling
```bash
# Set Round-Robin algorithm with 2.5 second time slice
scheduler round_robin 2.5

# Add jobs with different execution times
schedule task1 5 3.0    # Priority 5, needs 3 seconds
schedule task2 5 5.0    # Priority 5, needs 5 seconds  
schedule task3 5 2.0    # Priority 5, needs 2 seconds

# Check scheduler status
scheduler

# Expected behavior: Each job gets 2.5 seconds, then moves to end of queue
# Task3 completes first (2.0s < 2.5s), Task1 and Task2 continue
```

#### Priority-Based Scheduling
```bash
# Set Priority algorithm
scheduler priority

# Add jobs with different priorities
schedule high_priority_task 1 3.0    # Highest priority (1)
schedule medium_priority_task 5 4.0  # Medium priority (5)
schedule low_priority_task 10 2.0    # Lowest priority (10)

# Expected behavior: Jobs run in priority order (1, 5, 10)
```

#### Preemption Example
```bash
# Set Priority algorithm
scheduler priority

# Add a long-running low priority job
schedule long_task 10 10.0

# Wait a moment, then add a high priority job
schedule urgent_task 1 2.0

# Expected behavior: urgent_task preempts long_task and runs first
```

#### Advanced Job Creation
```bash
# Add job with all parameters
addjob my_task 3 7.5 true    # Priority 3, 7.5 seconds, background

# Check job status
jobs

# Monitor scheduler
scheduler
```

## Scheduling Algorithm Details

### Round-Robin Scheduling
- **Time Slice**: Configurable quantum (default: 2.0 seconds)
- **Queue Management**: FIFO queue with time slice enforcement
- **Process Switching**: Automatic after time slice expires
- **Completion**: Process removed when total time is reached
- **Fairness**: Equal time allocation regardless of priority

### Priority-Based Scheduling
- **Priority Levels**: 1 (highest) to 10 (lowest)
- **Queue Management**: Priority queue with FCFS for same priority
- **Preemption**: Higher priority jobs interrupt lower priority ones
- **Process Switching**: Immediate when higher priority job arrives
- **Completion**: Process runs to completion unless preempted

### Process Simulation
- **Execution Time**: Simulated using `time.sleep()`
- **Status Tracking**: Real-time updates (Waiting, Running, Done)
- **Thread Safety**: Thread-safe operations with locks
- **Performance Metrics**: Execution time, waiting time tracking

## Architecture

### Enhanced Modular Design
- **Shell**: Main command loop and user interaction
- **CommandParser**: Input tokenization and validation
- **CommandHandler**: Built-in command implementations (enhanced with scheduling)
- **JobManager**: Background process and job control (enhanced with scheduler integration)
- **Scheduler**: NEW: Process scheduling algorithms and queue management
- **Types**: Enhanced data structures for jobs and scheduling

### Threading Model
- **Main Thread**: Shell command processing and user interaction
- **Scheduler Thread**: Background process scheduling and execution
- **Thread Safety**: Lock-based synchronization for shared resources

### Error Handling
Comprehensive error handling for:
- Invalid commands and arguments
- File/directory permissions
- Process management errors
- Input validation errors
- Scheduling algorithm errors
- Thread synchronization errors

## Testing

### Built-in Testing
```bash
# Test scheduling features
python3 test_scheduling.py

# Test within shell
scheduler round_robin 1.0
schedule test1 5 2.0
schedule test2 5 3.0
scheduler
```

### Manual Testing Scenarios
```bash
# Test Round-Robin with different time slices
scheduler round_robin 1.0
schedule task1 5 5.0
schedule task2 5 3.0
scheduler round_robin 3.0

# Test Priority preemption
scheduler priority
schedule long_task 10 10.0
# Wait 2 seconds, then in another terminal or session:
schedule urgent_task 1 2.0

# Test job management with scheduled jobs
addjob my_job 3 5.0 true
jobs
fg 1  # Should show note about scheduled job
```

## Future Deliverables

### Deliverable 3: Memory Management & Synchronization
- Paging system with FIFO and LRU page replacement
- Process synchronization with mutexes/semaphores
- Classical synchronization problems (Producer-Consumer, Dining Philosophers)

### Deliverable 4: Integration & Security
- Command piping support
- User authentication system
- File permissions and access control
- Complete system integration

## Performance Considerations

- **Thread Safety**: All scheduler operations are thread-safe
- **Memory Efficiency**: Minimal memory overhead for process tracking
- **CPU Usage**: Efficient sleep-based simulation
- **Scalability**: Supports multiple concurrent processes
- **Real-time Updates**: Immediate status updates and process switching

## Troubleshooting

### Common Issues
1. **Scheduler not starting**: Check if jobs are added to the scheduler
2. **Processes not switching**: Verify time slice configuration
3. **Priority not working**: Ensure priority values are 1-10 (lower = higher priority)
4. **Jobs not completing**: Check total time needed vs time slice

### Debug Mode
```bash
python3 main.py --debug
```

## Contributing

This project demonstrates advanced operating system concepts including:
- Process scheduling algorithms
- Thread management and synchronization
- Job control and process management
- Command-line interface design
- Error handling and validation

The implementation serves as a foundation for understanding operating system internals and can be extended with additional features.