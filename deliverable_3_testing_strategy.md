# Deliverable 3 Testing Guide

## Memory Management and Process Synchronization

This guide provides comprehensive testing instructions for all Deliverable 3 requirements with expected outputs.

## Configuration

**Memory Manager Default Settings:**

- Default frame count: 12 frames (configurable)
- Default page replacement algorithm: FIFO
- Frame indexing: 0-11

---

## 🚀 Quick Start - Automated Testing

### Memory Management Tests

```bash
python3 main.py --test-memory
```

**What it does:**

- Tests FIFO page replacement algorithm with 4 frames
- Tests LRU page replacement with locality of reference
- Creates processes and simulates page allocation
- Demonstrates page faults vs page hits
- Shows page replacement in action

**Expected Output:**

```
=== TESTING MEMORY MANAGEMENT FEATURES ===

✓ Testing FIFO Algorithm:
  - Creating processes and allocating pages...
    ✓ Page allocated: P1:Pg0 -> Frame 0
    ✓ Page allocated: P2:Pg0 -> Frame 1
    ✓ Page allocated: P1:Pg1 -> Frame 2
    ✓ Page allocated: P2:Pg1 -> Frame 3
    ✓ Page allocated: P1:Pg2 -> Frame 0
    ✓ Page allocated: P2:Pg2 -> Frame 1
    ✓ Page allocated: P1:Pg3 -> Frame 2
    ✓ Page allocated: P2:Pg3 -> Frame 3
  - FIFO Results: 8 faults, 0 hits

✓ Testing LRU Algorithm:
  - Testing LRU with locality of reference...
    Access 1: Page 0 -> ✓ Page allocated: P3:Pg0 -> Frame 0
    Access 2: Page 1 -> ✓ Page allocated: P3:Pg1 -> Frame 1
    Access 3: Page 2 -> ✓ Page allocated: P3:Pg2 -> Frame 2
    Access 4: Page 0 -> ✓ Page hit: P3:Pg0
    Access 5: Page 1 -> ✓ Page hit: P3:Pg1
    Access 6: Page 3 -> ✓ Page allocated: P3:Pg3 -> Frame 3
    Access 7: Page 4 -> ✓ Page allocated: P3:Pg4 -> Frame 2
    Access 8: Page 0 -> ✓ Page hit: P3:Pg0
    Access 9: Page 1 -> ✓ Page hit: P3:Pg1
    Access 10: Page 5 -> ✓ Page allocated: P3:Pg5 -> Frame 3
  - LRU Results: 6 faults, 4 hits

✓ Memory management tests completed successfully!
```

### Process Synchronization Tests

```bash
python3 main.py --test-sync
```

**What it does:**

- Tests mutex creation, acquisition, and release
- Tests semaphore operations with multiple permits
- Runs Producer-Consumer simulation with statistics
- Runs Dining Philosophers with deadlock prevention
- Shows real-time status updates and final results

**Expected Output:**

```
=== TESTING SYNCHRONIZATION FEATURES ===

✓ Testing Synchronization Primitives:
  - Created mutex 'test_mutex'
  - Acquired mutex: True
  - Released mutex: True
  - Created semaphore 'test_sem' with value 3
  - Acquired semaphore 1: True
  - Acquired semaphore 2: True
  - Successfully acquired 2 semaphore permits
  - Synchronization status: 1 mutexes, 1 semaphores

✓ Testing Producer-Consumer Problem:
  - Starting Producer-Consumer simulation...
  - Buffer: 1/3
  - Items Produced: 8
  - Items Consumed: 7
  - Producer Waits: 2 (times producers waited for space)
  - Consumer Waits: 4 (times consumers waited for items)
  - Active: 2 producers, 2 consumers
  - Running: True
  - Final Results After Stop:
    * Total Produced: 10
    * Total Consumed: 10
    * Total Producer Waits: 3
    * Total Consumer Waits: 8
  - Producer-Consumer stopped

✓ Testing Dining Philosophers Problem:
  - Starting Dining Philosophers simulation...
  - Mid-simulation status (after 2 seconds):
    * Philosophers: 3
    * Current States: P0:eating, P1:thinking, P2:hungry
    * Meals So Far: 2
    * Deadlock Preventions: 1
    * Still Running: True
  - Final Results After Completion:
    * Total Meals Eaten: 6
    * Average Meals per Philosopher: 2.0
    * Individual Meals: P0:3, P1:2, P2:1
    * Total Deadlock Preventions: 3
    * Final States: P0:thinking, P1:eating, P2:thinking
    * Active Threads: 0
  - Dining Philosophers stopped

✓ Synchronization tests completed successfully!
```

---

## 🧪 Manual Testing Guide

### 1. Basic Paging System - Memory Allocation/Deallocation

```bash
# Start shell
python3 main.py

# Create processes with different memory requirements
memory create WebBrowser 6
memory create TextEditor 4
memory create MediaPlayer 8

# Check initial memory status
memory status

# Allocate pages for processes
memory alloc 1 0    # Allocate page 0 for process 1
memory alloc 1 1    # Allocate page 1 for process 1
memory alloc 2 0    # Allocate page 0 for process 2

# Check memory status after allocation
memory status

# Deallocate a process
memory dealloc 1

# Check final memory status
memory status
```

**Expected Output:**

```
✓ Process 1 (WebBrowser) created with 6 pages needed
✓ Process 2 (TextEditor) created with 4 pages needed
✓ Process 3 (MediaPlayer) created with 8 pages needed

=== MEMORY MANAGEMENT STATUS (NEW) ===
Total Frames: 12
Used Frames: 0
Free Frames: 12
✓ Memory Utilization: 0.0%
✓ Page Replacement: FIFO

=== PAGING STATISTICS (NEW) ===
✓ Page Faults: 0
✓ Page Hits: 0
✓ Page Replacements: 0
✓ Hit Ratio: 0.0%
Active Processes: 3

✓ Page allocated: P1:Pg0 -> Frame 0
✓ Page allocated: P1:Pg1 -> Frame 1
✓ Page allocated: P2:Pg0 -> Frame 2

=== MEMORY MANAGEMENT STATUS (NEW) ===
Total Frames: 12
Used Frames: 3
Free Frames: 9
✓ Memory Utilization: 25.0%
...
=== ACTIVE PROCESSES (NEW) ===
PID 1 (WebBrowser): 2/6 pages
PID 2 (TextEditor): 1/4 pages
PID 3 (MediaPlayer): 0/8 pages

✓ Process 1 deallocated, freed 2 pages
```

### 2. Page Fault Handling

```bash
# Create a process
memory create TestApp 5

# Try to allocate pages (will show page faults)
memory alloc 1 0    # First access - will cause page fault
memory alloc 1 0    # Second access - will be page hit
memory alloc 1 1    # New page - will cause page fault

# Check statistics to see page faults vs hits
memory status
```

**Expected Output:**

```
✓ Process 1 (TestApp) created with 5 pages needed
✓ Page allocated: P1:Pg0 -> Frame 0
✓ Page hit: P1:Pg0
✓ Page allocated: P1:Pg1 -> Frame 1

=== PAGING STATISTICS (NEW) ===
✓ Page Faults: 2
✓ Page Hits: 1
✓ Page Replacements: 0
✓ Hit Ratio: 33.3%
```

### 3. FIFO Page Replacement Algorithm

```bash
memory algorithm fifo # Set algorithm to FIFO
memory create LargeApp 15 # Create process with 15 pages (0-14)

# Phase 1: Fill all 12 available frames
memory alloc 1 0  # Frame 0
memory alloc 1 1  # Frame 1
memory alloc 1 2  # Frame 2
memory alloc 1 3  # Frame 3
memory alloc 1 4  # Frame 4
memory alloc 1 5  # Frame 5
memory alloc 1 6  # Frame 6e
memory alloc 1 7  # Frame 7
memory alloc 1 8  # Frame 8
memory alloc 1 9  # Frame 9
memory alloc 1 10 # Frame 10
memory alloc 1 11 # Frame 11 (all frames now full since default frame is 12)

# Phase 2: Force FIFO replacements
memory alloc 1 12 # Should replace Page 0 (oldest in FIFO queue)
memory alloc 1 13 # Should replace Page 1 (next oldest in FIFO queue)
memory alloc 1 14 # Should replace Page 2 (next oldest in FIFO queue)

# Phase 3: Test what got replaced vs what's still in memory
memory alloc 1 0  # Should be page fault (was replaced)
memory alloc 1 1  # Should be page fault (was replaced)
memory alloc 1 2  # Should be page fault (was replaced)
memory alloc 1 11 # Should be page hit (still in memory)
memory alloc 1 12 # Should be page hit (recently loaded)

# Check statistics
memory status
```

**Expected Output:**

```
✓ Page replacement algorithm set to FIFO
✓ Process 1 (LargeApp) created with 15 pages needed
✓ Page allocated: P1:Pg0 -> Frame 0
✓ Page allocated: P1:Pg1 -> Frame 1
✓ Page allocated: P1:Pg2 -> Frame 2
✓ Page allocated: P1:Pg3 -> Frame 3
✓ Page allocated: P1:Pg4 -> Frame 4
✓ Page allocated: P1:Pg5 -> Frame 5
✓ Page allocated: P1:Pg6 -> Frame 6
✓ Page allocated: P1:Pg7 -> Frame 7
✓ Page allocated: P1:Pg8 -> Frame 8
✓ Page allocated: P1:Pg9 -> Frame 9
✓ Page allocated: P1:Pg10 -> Frame 10
✓ Page allocated: P1:Pg11 -> Frame 11
✓ Page allocated: P1:Pg12 -> Frame 0    # FIFO replacement (Page 0 evicted)
✓ Page allocated: P1:Pg13 -> Frame 1    # FIFO replacement (Page 1 evicted)
✓ Page allocated: P1:Pg14 -> Frame 2    # FIFO replacement (Page 2 evicted)
✓ Page allocated: P1:Pg0 -> Frame 3     # Page fault (Page 0 was replaced)
✓ Page allocated: P1:Pg1 -> Frame 4     # Page fault (Page 1 was replaced)
✓ Page allocated: P1:Pg2 -> Frame 5     # Page fault (Page 2 was replaced)
✓ Page hit: P1:Pg11                     # Still in memory (Frame 11)
✓ Page hit: P1:Pg12                     # Still in memory (Frame 0)

=== PAGING STATISTICS (NEW) ===
✓ Page Faults: 18
✓ Page Hits: 2
✓ Page Replacements: 6
✓ Hit Ratio: 10.0%
```

### 4. LRU Page Replacement Algorithm

```bash
# Set algorithm to LRU
memory algorithm lru

# Create process
memory create LRUTest 8

# Create access pattern with locality of reference
memory alloc 1 0    # Load page 0
memory alloc 1 1    # Load page 1
memory alloc 1 2    # Load page 2
memory alloc 1 0    # Access page 0 again (updates LRU)
memory alloc 1 3    # Load page 3
memory alloc 1 4    # Load page 4
memory alloc 1 5    # Should replace least recently used
memory alloc 1 0    # Should be hit if page 0 still in memory

# Check statistics
memory status
```

**Expected Output:**

```
✓ Page replacement algorithm set to LRU
✓ Process 1 (LRUTest) created with 8 pages needed
✓ Page allocated: P1:Pg0 -> Frame 0
✓ Page allocated: P1:Pg1 -> Frame 1
✓ Page allocated: P1:Pg2 -> Frame 2
✓ Page hit: P1:Pg0                    # LRU updated
✓ Page allocated: P1:Pg3 -> Frame 3
✓ Page allocated: P1:Pg4 -> Frame 4
✓ Page allocated: P1:Pg5 -> Frame 2   # LRU replacement (page 2 was least recent)
✓ Page hit: P1:Pg0                    # Page 0 still in memory

=== PAGING STATISTICS (NEW) ===
✓ Page Faults: 6
✓ Page Hits: 2
✓ Page Replacements: 1
✓ Hit Ratio: 25.0%
```

### 5. Producer-Consumer Problem

```bash
sync prodcons start 2 3 # Start Producer-Consumer with 2 producers, 3 consumers
sync prodcons status # Check status while running
sync prodcons status # Let it run for a while, then check again
sync prodcons stop # Stop the simulation
sync prodcons status # Check final results
```

**Expected Output:**

```
✓ Producer-Consumer started: 2 producers, 3 consumers

Producer-Consumer Status:
Running: True
Buffer: 2/5
✓ Produced: 12
✓ Consumed: 10
Active Producers: 2
Active Consumers: 3
Producer Waits: 1
Consumer Waits: 5

Producer-Consumer Status:
Running: True
Buffer: 0/5
✓ Produced: 20
✓ Consumed: 20
Active Producers: 0
Active Consumers: 0
Producer Waits: 3
Consumer Waits: 12

✓ Producer-Consumer stopped

Producer-Consumer Status:
Running: False
Buffer: 0/5
✓ Produced: 20
✓ Consumed: 20
Active Producers: 0
Active Consumers: 0
Producer Waits: 3
Consumer Waits: 12
```

### 6. Dining Philosophers Problem

```bash
# Start Dining Philosophers with 5 philosophers for 20 seconds
sync philosophers start 5 20
sync philosophers status # Check status while running
sync philosophers stop # Wait for completion or stop manually
sync philosophers status # Check final results showing deadlock prevention
```

**Expected Output:**

```
✓ Dining Philosophers started: 5 philosophers for 20s

Dining Philosophers Status:
Running: True
Philosophers: 5
States: P0:eating, P1:thinking, P2:hungry, P3:eating, P4:thinking
✓ Total Meals: 8
✓ Average Meals: 1.6
Meals: P0:2, P1:1, P2:1, P3:3, P4:1
✓ Deadlock Preventions: 4
Active Threads: 5

# After 20 seconds or manual stop:

Dining Philosophers Status:
Running: False
Philosophers: 5
States: P0:thinking, P1:eating, P2:thinking, P3:thinking, P4:hungry
✓ Total Meals: 26
✓ Average Meals: 5.2
Meals: P0:6, P1:5, P2:6, P3:4, P4:5
✓ Deadlock Preventions: 11
Active Threads: 0
```

### 7. Comprehensive Integration Test

```bash
# Test both memory and sync together
memory create SyncApp 6
memory algorithm lru
sync mutex create app_mutex
sync prodcons start 2 2

# Check both systems running
memory status
sync status

# Stress test
memory test random
sync philosophers start 3 10

# Final comprehensive status
memory status
sync status

# Cleanup
sync prodcons stop
sync philosophers stop
```

**Expected Output:**

```
✓ Process 1 (SyncApp) created with 6 pages needed
✓ Page replacement algorithm set to LRU
✓ Mutex 'app_mutex' created
✓ Producer-Consumer started: 2 producers, 2 consumers

=== MEMORY MANAGEMENT STATUS (NEW) ===
Total Frames: 12
Used Frames: 0
Free Frames: 12
✓ Memory Utilization: 0.0%
✓ Page Replacement: LRU
...

=== PROCESS SYNCHRONIZATION STATUS (NEW) ===
✓ Mutexes: 1
✓ Semaphores: 0
✓ Lock Acquisitions: 0
✓ Lock Waits: 0
Shared Resources: 0

=== PRODUCER-CONSUMER STATUS (NEW) ===
Running: True
Buffer: 1/5
✓ Items Produced: 8
✓ Items Consumed: 7
Active Producers: 2
Active Consumers: 2

✓ Testing random memory access pattern...
Access 1: ✓ Page allocated: P2:Pg3 -> Frame 0
Access 2: ✓ Page allocated: P2:Pg1 -> Frame 1
...

✓ Dining Philosophers started: 3 philosophers for 10s

# Final status shows both systems working together
✓ Producer-Consumer stopped
✓ Dining Philosophers stopped
```

---

## 📊 Key Success Indicators

### Memory Management

- **Page Faults**: Should occur on first access to each page
- **Page Hits**: Should occur on subsequent access to same page
- **Page Replacements**: Should occur when memory is full
- **Hit Ratio**: Should improve with locality of reference
- **Clean Deallocation**: All pages freed when process deallocated

### Process Synchronization

- **No Deadlocks**: Dining philosophers should complete without hanging
- **Race Prevention**: Producer-Consumer buffer should never have invalid states
- **Fair Access**: All philosophers should get some meals
- **Proper Cleanup**: All threads should terminate cleanly (Active Threads: 0)
- **Statistics**: Waits and preventions show synchronization working

### Integration

- **Both Systems Running**: Memory and sync should work simultaneously
- **No Interference**: One system shouldn't break the other
- **Resource Management**: Clean startup and shutdown of all components
- **Error Handling**: Graceful handling of invalid inputs and edge cases

---

## 🐛 Troubleshooting

### Common Issues

1. **"Command not found"**: Ensure you're in the shell (`python3 main.py`) not system terminal
2. **"Process not found"**: Use `memory status` to check existing process IDs
3. **Tests hanging**: Use Ctrl+C to interrupt, then `exit` to quit shell properly
4. **Import errors**: Ensure all `.py` files are in the same directory

### Expected Behaviors

- Page replacements will vary based on access patterns and timing
- Dining philosophers meal counts depend on random timing and duration
- Producer-Consumer results vary with number of producers/consumers and timing
- Some variation in statistics is normal and expected

This testing guide demonstrates all Deliverable 3 requirements with realistic expected outputs! 🎯