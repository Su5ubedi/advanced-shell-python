#!/usr/bin/env python3
"""
main.py - Entry point for Advanced Shell Simulation
Updated for Deliverable 3: Memory Management and Process Synchronization
"""

import argparse
import sys
from shell import Shell


def show_version():
    """Show version information"""
    print("Advanced Shell Simulation")
    print("Version: 3.0.0 (Deliverable 3)")
    print("Build: Development")
    print()
    print("Features:")
    print("- Basic shell functionality (Deliverable 1)")
    print("- Built-in commands")
    print("- Process management")
    print("- Job control")
    print("- Process scheduling algorithms (Deliverable 2)")
    print("  * Round-Robin Scheduling")
    print("  * Priority-Based Scheduling")
    print("- Performance metrics")
    print("- Real-time process monitoring")
    print("✓ Memory management with paging (Deliverable 3 - NEW)")
    print("  * FIFO and LRU page replacement algorithms")
    print("  * Page fault handling and tracking")
    print("  * Memory overflow simulation")
    print("✓ Process synchronization (Deliverable 3 - NEW)")
    print("  * Mutexes and semaphores")
    print("  * Producer-Consumer problem")
    print("  * Dining Philosophers problem")
    print("  * Race condition prevention")
    print("  * Deadlock avoidance")


def show_help():
    """Show help information"""
    print("Advanced Shell Simulation - Deliverable 3")
    print()
    print("Usage:")
    print("  python3 main.py [options]")
    print()
    print("Options:")
    print("  --version    Show version information")
    print("  --help       Show this help message")
    print("  --debug      Enable debug mode")
    print("  --test-memory    Run memory management tests (NEW)")
    print("  --test-sync      Run synchronization tests (NEW)")
    print()
    print("Once started, type 'help' for available shell commands")
    print()
    print("Quick Start - Process Scheduling:")
    print("  1. Configure algorithm:    scheduler config rr 2")
    print("  2. Add processes:          scheduler addprocess task1 5")
    print("  3. Start scheduler:        scheduler start")
    print("  4. Monitor status:         scheduler status")
    print("  5. View metrics:           scheduler metrics")
    print()
    print("✓ Quick Start - Memory Management (NEW):")
    print("  1. Create process:         memory create webapp 8")
    print("  2. Allocate pages:         memory alloc 1 0")
    print("  3. Set algorithm:          memory algorithm lru")
    print("  4. View status:            memory status")
    print("  5. Test patterns:          memory test random")
    print()
    print("✓ Quick Start - Synchronization (NEW):")
    print("  1. Create mutex:           sync mutex create mylock")
    print("  2. Start Producer-Consumer: sync prodcons start 2 3")
    print("  3. Start Dining Philosophers: sync philosophers start 5")
    print("  4. View status:            sync status")


def test_memory_management():
    """Test memory management features (NEW)"""
    print("=== TESTING MEMORY MANAGEMENT FEATURES ===")
    print()

    try:
        from memory_manager import MemoryManager

        # Test 1: Memory Usage Tracking per Process
        print("✓ Testing Memory Usage Tracking per Process:")
        mm = MemoryManager(total_frames=6)  # Small frames to force overflow

        # Create multiple processes with different memory needs
        pid1 = mm.create_process("WebBrowser", 4)
        pid2 = mm.create_process("TextEditor", 3)
        pid3 = mm.create_process("MediaPlayer", 5)

        print("  - Created processes with different memory requirements")
        print(f"    * WebBrowser (PID {pid1}): needs 4 pages")
        print(f"    * TextEditor (PID {pid2}): needs 3 pages")
        print(f"    * MediaPlayer (PID {pid3}): needs 5 pages")

        # Allocate pages for each process
        print("  - Allocating pages for each process...")
        mm.allocate_page(pid1, 0)  # WebBrowser page 0
        mm.allocate_page(pid1, 1)  # WebBrowser page 1
        mm.allocate_page(pid2, 0)  # TextEditor page 0
        mm.allocate_page(pid3, 0)  # MediaPlayer page 0
        mm.allocate_page(pid3, 1)  # MediaPlayer page 1

        status = mm.get_status()
        print(f"  - Memory Usage: {status['used_frames']}/{status['total_frames']} frames ({status['utilization']:.1f}%)")
        print(f"  - Per-Process Allocation:")
        for pid, process in mm.processes.items():
            allocated = len([p for p in process.page_table.values() if p is not None])
            print(f"    * PID {pid} ({process.name}): {allocated}/{process.pages_needed} pages allocated")
        print()

        # Test 2: Memory Overflow Simulation
        print("✓ Testing Memory Overflow Scenarios:")
        mm.set_algorithm("fifo")
        print("  - Set algorithm to FIFO for overflow testing")

        # Force memory overflow by allocating more pages than available frames
        print("  - Forcing memory overflow (6 frames available, allocating more)...")
        overflow_results = []

        # Fill remaining frame
        success, msg = mm.allocate_page(pid3, 2)  # Frame 6 (last free frame)
        overflow_results.append(f"    Frame 6: {msg}")

        # Now force page replacements
        success, msg = mm.allocate_page(pid1, 2)  # Should trigger replacement
        overflow_results.append(f"    Overflow 1: {msg}")

        success, msg = mm.allocate_page(pid2, 1)  # Should trigger another replacement
        overflow_results.append(f"    Overflow 2: {msg}")

        success, msg = mm.allocate_page(pid3, 3)  # Should trigger another replacement
        overflow_results.append(f"    Overflow 3: {msg}")

        for result in overflow_results:
            print(result)

        status = mm.get_status()
        print(f"  - Memory overflow results:")
        print(f"    * Total page faults: {status['page_faults']}")
        print(f"    * Page replacements triggered: {status['replacements']}")
        print(f"    * Memory utilization: {status['utilization']:.1f}% (should be 100%)")
        print()

        # Test 3: Page Fault Tracking
        print("✓ Testing Page Fault Tracking:")
        print("  - Testing page fault vs page hit patterns...")

        # Access same pages multiple times to show fault vs hit tracking
        test_accesses = [
            (pid1, 0, "First access (should be hit - already allocated)"),
            (pid1, 3, "New page (should be fault)"),
            (pid1, 0, "Repeat access (should be hit)"),
            (pid2, 2, "New page (should be fault)"),
            (pid1, 3, "Repeat access (should be hit)"),
        ]

        for pid, page_num, description in test_accesses:
            old_faults = mm.page_faults
            old_hits = mm.page_hits
            success, msg = mm.allocate_page(pid, page_num)
            new_faults = mm.page_faults
            new_hits = mm.page_hits

            fault_occurred = new_faults > old_faults
            hit_occurred = new_hits > old_hits
            result_type = "FAULT" if fault_occurred else "HIT" if hit_occurred else "ERROR"

            print(f"    * {description}: {result_type}")

        final_status = mm.get_status()
        print(f"  - Final page fault statistics:")
        print(f"    * Total page faults: {final_status['page_faults']}")
        print(f"    * Total page hits: {final_status['page_hits']}")
        print(f"    * Hit ratio: {final_status['hit_ratio']:.1f}%")
        print()

        # Test 4: FIFO vs LRU Comparison
        print("✓ Testing FIFO vs LRU Page Replacement:")

        # Test FIFO
        mm_fifo = MemoryManager(total_frames=4)
        mm_fifo.set_algorithm("fifo")
        pid_fifo = mm_fifo.create_process("FIFOTest", 6)

        print("  - FIFO Algorithm Test:")
        fifo_pattern = [0, 1, 2, 3, 0, 1, 4, 5]  # Will cause replacements
        for i, page_num in enumerate(fifo_pattern):
            success, msg = mm_fifo.allocate_page(pid_fifo, page_num)

        fifo_status = mm_fifo.get_status()
        print(f"    * FIFO Results: {fifo_status['page_faults']} faults, {fifo_status['page_hits']} hits, {fifo_status['replacements']} replacements")

        # Test LRU
        mm_lru = MemoryManager(total_frames=4)
        mm_lru.set_algorithm("lru")
        pid_lru = mm_lru.create_process("LRUTest", 6)

        print("  - LRU Algorithm Test:")
        lru_pattern = [0, 1, 2, 3, 0, 1, 4, 5]  # Same pattern, different results
        for i, page_num in enumerate(lru_pattern):
            success, msg = mm_lru.allocate_page(pid_lru, page_num)

        lru_status = mm_lru.get_status()
        print(f"    * LRU Results: {lru_status['page_faults']} faults, {lru_status['page_hits']} hits, {lru_status['replacements']} replacements")

        print(f"  - Algorithm Comparison:")
        print(f"    * FIFO hit ratio: {fifo_status['hit_ratio']:.1f}%")
        print(f"    * LRU hit ratio: {lru_status['hit_ratio']:.1f}%")
        print(f"    * LRU is {'better' if lru_status['hit_ratio'] > fifo_status['hit_ratio'] else 'same as'} FIFO for this pattern")
        print()

        print("✓ Memory management tests completed successfully!")
        print("✓ Demonstrated: Per-process tracking, memory overflow, page fault tracking, algorithm comparison")

    except ImportError as e:
        print(f"Error: Could not import memory management modules: {e}")
    except Exception as e:
        print(f"Error during memory tests: {e}")


def test_synchronization():
    """Test synchronization features (NEW)"""
    print("=== TESTING SYNCHRONIZATION FEATURES ===")
    print()

    try:
        from process_sync import ProcessSynchronizer, ProducerConsumer, DiningPhilosophers
        import time
        import threading

        # Test basic synchronization primitives
        print("✓ Testing Synchronization Primitives:")
        sync = ProcessSynchronizer()

        # Test mutex
        sync.create_mutex("test_mutex")
        print("  - Created mutex 'test_mutex'")

        success = sync.acquire_mutex("test_mutex")
        print(f"  - Acquired mutex: {success}")

        if success:
            success = sync.release_mutex("test_mutex")
            print(f"  - Released mutex: {success}")
        else:
            print("  - Skipping release (acquire failed)")

        # Test semaphore
        sync.create_semaphore("test_sem", 3)
        print("  - Created semaphore 'test_sem' with value 3")

        acquired_count = 0
        for i in range(2):
            success = sync.acquire_semaphore("test_sem")
            if success:
                acquired_count += 1
            print(f"  - Acquired semaphore {i+1}: {success}")

        print(f"  - Successfully acquired {acquired_count} semaphore permits")

        status = sync.get_status()
        print(f"  - Synchronization status: {status['mutexes']} mutexes, {status['semaphores']} semaphores")
        print()

        # Test Producer-Consumer
        print("✓ Testing Producer-Consumer Problem:")
        pc = ProducerConsumer(buffer_size=3)
        print("Starting Producer-Consumer simulation... (Number of producers: 2, consumers: 3)")

        pc.start(num_producers=2, num_consumers=3, duration=5)
        time.sleep(10)  # Let it run briefly

        pc_status = pc.get_status()
        print(f"  - Buffer: {pc_status['current_buffer']}/{pc_status['buffer_size']}")
        print(f"  - Items Produced: {pc_status['items_produced']}")
        print(f"  - Items Consumed: {pc_status['items_consumed']}")
        print(f"  - Producer Waits: {pc_status['producer_waits']} (times producers waited for space)")
        print(f"  - Consumer Waits: {pc_status['consumer_waits']} (times consumers waited for items)")
        print(f"  - Produced: {pc_status['items_produced']}, Consumed: {pc_status['items_consumed']}")
        print(f"  - Active: {pc_status['active_producers']} producers, {pc_status['active_consumers']} consumers")

        pc.stop()
        print("Producer-Consumer stopped")
        print()

        # Test Dining Philosophers
        print("✓ Testing Dining Philosophers Problem: (Number of philosophers: 5) (Time duration: 3 seconds)")
        dp = DiningPhilosophers(num_philosophers=5)
        print("Starting Dining Philosophers simulation...")

        # Run in background thread for brief test
        def run_philosophers():
            dp.start(duration=3)

        phil_thread = threading.Thread(target=run_philosophers)
        phil_thread.daemon = True
        phil_thread.start()

        time.sleep(3)  # Let it run briefly

        dp_status = dp.get_status()
        print(f"  - Philosophers: {dp_status['num_philosophers']}")
        print(f"  - Total meals eaten: {dp_status['total_meals']}")
        print(f"  - Deadlock preventions: {dp_status['deadlock_preventions']}")
        print(f"  - States: {', '.join(dp_status['states'])}")

        dp.stop()
        print("Dining Philosophers stopped")
        print()

        print("✓ Synchronization tests completed successfully!")

    except ImportError as e:
        print(f"Error: Could not import synchronization modules: {e}")
    except Exception as e:
        print(f"Error during synchronization tests: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Shell Simulation - Deliverable 3",
        add_help=False  # We'll handle help ourselves
    )

    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information'
    )

    parser.add_argument(
        '--help',
        action='store_true',
        help='Show help information'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    # Deliverable 3: NEW test options
    parser.add_argument(
        '--test-memory',
        action='store_true',
        help='Run memory management tests and exit (NEW)'
    )

    parser.add_argument(
        '--test-sync',
        action='store_true',
        help='Run synchronization tests and exit (NEW)'
    )

    args = parser.parse_args()

    if args.version:
        show_version()
        return

    if args.help:
        show_help()
        return

    # Deliverable 3: NEW - Handle test options
    if args.test_memory:
        test_memory_management()
        return

    if args.test_sync:
        test_synchronization()
        return

    # Create and start the shell
    try:
        shell = Shell()

        if args.debug:
            print("Debug mode enabled")
            print("Scheduler modules available:", hasattr(shell.command_handler, 'process_scheduler'))
            # Deliverable 3: NEW debug info
            print("Memory management available:", hasattr(shell.command_handler, 'memory_sync_commands'))
            if hasattr(shell.command_handler, 'memory_sync_commands'):
                memory_manager = shell.command_handler.get_memory_manager()
                synchronizer = shell.command_handler.get_synchronizer()
                print(f"  - Memory manager: {memory_manager.total_frames} frames")
                print(f"  - Synchronizer: initialized")

        # Run the shell
        shell.run()

    except KeyboardInterrupt:
        print("\nShell interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()