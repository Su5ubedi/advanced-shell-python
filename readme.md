# Advanced Shell Simulation - Deliverable 3

A custom shell implementation in Python that simulates Unix-like Operating System environment with process management, job control, advanced scheduling algorithms, **memory management with paging**, and **process synchronization**.

## Features

### Core Shell Features (Deliverable 1)
- **Built-in Commands**: `cd`, `pwd`, `echo`, `clear`, `ls`, `cat`, `mkdir`, `rmdir`, `rm`, `touch`, `kill`, `exit`, `help`
- **Process Management**: Foreground and background execution
- **Job Control**: `jobs`, `fg`, `bg`, `stop` commands
- **Signal handling and error management**

### Process Scheduling (Deliverable 2)
- **Round-Robin Scheduling**: Configurable time slice (default: 2.0 seconds)
- **Priority-Based Scheduling**: Preemptive priority queue (1=highest, 10=lowest)
- **Process Simulation**: Uses `time.sleep()` to simulate execution
- **Performance Metrics**: Execution time, waiting time tracking

### ✓ Memory Management (Deliverable 3 - NEW)
- **Paging System**: Fixed-size page frames with allocation/deallocation
- **Page Replacement Algorithms**:
  - **FIFO**: First-In-First-Out replacement
  - **LRU**: Least Recently Used replacement
- **Page Fault Handling**: Tracks faults, hits, and replacements
- **Memory Statistics**: Utilization, hit ratios, performance metrics

### ✓ Process Synchronization (Deliverable 3 - NEW)
- **Synchronization Primitives**: Mutexes and semaphores
- **Classical Problems**:
  - **Producer-Consumer**: Multi-threaded buffer synchronization
  - **Dining Philosophers**: Deadlock prevention with asymmetric fork acquisition
- **Race Condition Prevention**: Thread-safe operations with proper locking
- **Deadlock Avoidance**: Asymmetric resource acquisition strategy

## Project Structure

```
advanced-shell-python/
├── main.py                    # Entry point with command-line options
├── shell.py                   # Main shell class and command loop
├── command_handler.py         # Built-in command implementations (updated)
├── command_parser.py          # Command parsing and validation (updated)
├── job_manager.py             # Job control operations
├── scheduler.py               # Process scheduling algorithms
├── shell_types.py             # Data types and enums
├── memory_manager.py          # NEW: Memory management with paging
├── process_sync.py            # NEW: Process synchronization
├── shell_integration.py       # NEW: Integration of memory and sync features
└── README.md                  # This file
```

## Installation & Usage

### Requirements
- Python 3.8 or higher

### Running the Shell
```bash
# Start the shell
python3 main.py

# Command-line options
python3 main.py --help          # Show help
python3 main.py --version       # Show version info
python3 main.py --debug         # Enable debug mode
python3 main.py --test-memory   # Test memory management (NEW)
python3 main.py --test-sync     # Test synchronization (NEW)
```

## ✓ New Commands (Deliverable 3)

### Memory Management Commands
```bash
memory status                    # Show memory status and statistics
memory create <name> <pages>     # Create process with memory requirements
memory alloc <pid> <page_num>    # Allocate specific page for process
memory dealloc <pid>             # Deallocate all pages for process
memory algorithm <fifo|lru>      # Set page replacement algorithm
memory test <sequential|random>  # Run memory access pattern test
```

### Process Synchronization Commands
```bash
sync status                               # Show synchronization status
sync mutex <create|acquire|release> <n>    # Mutex operations
sync semaphore <create|acquire|release> <n> [value] # Semaphore operations
sync prodcons <start|stop|status> [producers] [consumers] # Producer-Consumer
sync philosophers <start|stop|status> [philosophers] # Dining Philosophers
```

## 🧪 Deliverable 3 Testing Guide

### Quick Tests
```bash
# Memory Management
memory create webapp 8          # Create process
memory alloc 1 0                # Allocate page (causes page fault)
memory alloc 1 0                # Second access (page hit)
memory algorithm lru            # Switch algorithms
memory status                   # View statistics

# Process Synchronization
sync mutex create mylock        # Create mutex
sync prodcons start 2 3         # Start Producer-Consumer
sync philosophers start 5       # Start Dining Philosophers
sync status                     # View all sync status
```

### Comprehensive Tests
```bash
# Test page replacement (memory overflow)
memory create LargeApp 10
memory test sequential          # Forces FIFO/LRU replacements

# Test race condition prevention
sync prodcons start 3 2         # Multiple producers/consumers
sync prodcons status            # Monitor safe buffer access

# Test deadlock prevention
sync philosophers start 5 20    # Long-running simulation
sync philosophers status        # Check deadlock preventions
```

### Automated Testing
```bash
python3 main.py --test-memory   # Automated memory tests
python3 main.py --test-sync     # Automated synchronization tests
```

## Example Usage

### Basic Shell Operations
```bash
ls -la
mkdir -p test/nested/path
echo "Hello World\nLine 2"
sleep 10 &                     # Background job
jobs                           # List jobs
fg 1                           # Bring to foreground
```

### Process Scheduling
```bash
scheduler round_robin 2.5      # Configure Round-Robin
schedule task1 5 3.0           # Add job with priority 5, 3s duration
scheduler start                # Start scheduling
scheduler status               # Monitor execution
```

### ✓ Memory Management (NEW)
```bash
memory create browser 6        # Create process needing 6 pages
memory alloc 1 0              # Allocate page 0 -> Page fault
memory alloc 1 1              # Allocate page 1 -> Page fault
memory alloc 1 0              # Access page 0 -> Page hit
memory algorithm lru          # Switch to LRU algorithm
memory status                 # View: 2 faults, 1 hit, 66.7% hit ratio
```

### ✓ Process Synchronization (NEW)
```bash
# Producer-Consumer Problem
sync prodcons start 2 3       # 2 producers, 3 consumers
sync prodcons status          # Monitor: items produced/consumed, waits
sync prodcons stop            # Stop simulation

# Dining Philosophers Problem
sync philosophers start 5     # 5 philosophers
sync philosophers status      # Monitor: meals eaten, deadlock preventions
# Auto-stops after 15 seconds
```

## Architecture

### Memory Management
- **Page Frames**: Fixed-size memory units (default: 12 frames)
- **Page Table**: Maps logical pages to physical frames per process
- **Replacement**: FIFO queue and LRU access tracking
- **Statistics**: Page faults, hits, replacements, utilization

### Process Synchronization
- **Thread Safety**: All operations use proper locking
- **Deadlock Prevention**: Asymmetric resource acquisition
- **Classical Problems**: Producer-Consumer and Dining Philosophers
- **Performance**: Lock acquisitions, waits, success rates tracked

### Integration
- **Minimal Changes**: New features cleanly integrated into existing shell
- **Backward Compatibility**: All existing commands work unchanged
- **Unified Interface**: Memory and sync commands follow same patterns
- **Error Handling**: Comprehensive validation and error messages

## Performance Features

- **Real-time Monitoring**: Live status updates for all subsystems
- **Statistics Tracking**: Comprehensive metrics for analysis
- **Thread Management**: Clean startup/shutdown of background processes
- **Resource Cleanup**: Proper deallocation and thread termination

## Development

Built with Python 3.8+ using:
- **Threading**: For concurrent process simulation
- **Collections**: OrderedDict for LRU tracking
- **Queue**: Thread-safe Producer-Consumer buffer
- **Time**: Process execution simulation
- **Enum**: Type-safe status and algorithm definitions

---

**Note**: This implementation demonstrates core operating system concepts including process management, memory management, and synchronization in a simulated environment suitable for educational purposes.