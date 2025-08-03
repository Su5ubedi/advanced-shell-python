#!/usr/bin/env python3
"""
shell_integration.py - Integration Module for Deliverable 3
Integrates memory management and process synchronization into the existing shell
✓ Memory management commands integration (NEW)
✓ Process synchronization commands integration (NEW)
✓ Classical synchronization problems demos (NEW)
✓ Minimal impact on existing shell structure (NEW)
"""

from memory_manager import MemoryManager
from process_sync import ProcessSynchronizer, ProducerConsumer, DiningPhilosophers
import threading
from typing import List


class MemorySyncCommands:
    """Command handler class for Deliverable 3 integration (NEW)"""

    def __init__(self):
        self.memory_manager = MemoryManager(total_frames=12)
        self.synchronizer = ProcessSynchronizer()
        self.producer_consumer = None
        self.dining_philosophers = None

    def handle_memory(self, args: List[str]) -> str:
        """Handle memory management commands (NEW)"""
        if not args:
            return self._memory_help()

        command = args[0].lower()

        if command == "status":
            return self._memory_status()
        elif command == "create" and len(args) >= 3:
            return self._create_process(args[1], int(args[2]))
        elif command == "alloc" and len(args) >= 3:
            return self._allocate_page(int(args[1]), int(args[2]))
        elif command == "dealloc" and len(args) >= 2:
            return self._deallocate_process(int(args[1]))
        elif command == "algorithm" and len(args) >= 2:
            return self._set_algorithm(args[1])
        elif command == "test" and len(args) >= 2:
            return self._test_memory(args[1])
        else:
            return self._memory_help()

    def handle_sync(self, args: List[str]) -> str:
        """Handle synchronization commands (NEW)"""
        if not args:
            return self._sync_help()

        command = args[0].lower()

        if command == "status":
            return self._sync_status()
        elif command == "mutex" and len(args) >= 3:
            return self._handle_mutex(args[1], args[2:])
        elif command == "semaphore" and len(args) >= 3:
            return self._handle_semaphore(args[1], args[2:])
        elif command == "prodcons":
            return self._handle_producer_consumer(args[1:])
        elif command == "philosophers":
            return self._handle_dining_philosophers(args[1:])
        else:
            return self._sync_help()

    def _memory_help(self) -> str:
        """Memory management help"""
        return """✓ Memory Management Commands:
memory status                     - Show memory status and statistics
memory create <name> <pages>      - Create process with pages needed
memory alloc <pid> <page_num>     - Allocate specific page for process
memory dealloc <pid>              - Deallocate all pages for process
memory algorithm <fifo|lru>       - Set page replacement algorithm
memory test <sequential|random>   - Run memory access pattern test"""

    def _sync_help(self) -> str:
        """Synchronization help"""
        return """✓ Process Synchronization Commands:
sync status                               - Show synchronization status
sync mutex <create|acquire|release> <name>    - Mutex operations
sync semaphore <create|acquire|release> <name> [value] - Semaphore operations
sync prodcons <start|stop|status> [producers] [consumers] - Producer-Consumer
sync philosophers <start|stop|status> [num_philosophers] - Dining Philosophers"""

    def _memory_status(self) -> str:
        """Get memory status"""
        status = self.memory_manager.get_status()

        result = f"""=== MEMORY MANAGEMENT STATUS ===
Total Frames: {status['total_frames']}
Used Frames: {status['used_frames']}
Free Frames: {status['free_frames']}
✓ Memory Utilization: {status['utilization']:.1f}%
✓ Page Replacement: {status['algorithm'].upper()}

=== PAGING STATISTICS===
✓ Page Faults: {status['page_faults']}
✓ Page Hits: {status['page_hits']}
✓ Page Replacements: {status['replacements']}
✓ Hit Ratio: {status['hit_ratio']:.1f}%
Active Processes: {status['processes']}"""

        if self.memory_manager.processes:
            result += "\n\n=== ACTIVE PROCESSES==="
            for pid, process in self.memory_manager.processes.items():
                allocated = len([p for p in process.page_table.values() if p is not None])
                result += f"\nPID {pid} ({process.name}): {allocated}/{process.pages_needed} pages"

        return result

    def _create_process(self, name: str, pages: int) -> str:
        """Create a new process (NEW)"""
        try:
            pid = self.memory_manager.create_process(name, pages)
            return f"✓ Process {pid} ({name}) created with {pages} pages needed"
        except Exception as e:
            return f"Error creating process: {e}"

    def _allocate_page(self, pid: int, page_num: int) -> str:
        """Allocate a page for a process (NEW)"""
        try:
            success, message = self.memory_manager.allocate_page(pid, page_num)
            return message
        except Exception as e:
            return f"Error allocating page: {e}"

    def _deallocate_process(self, pid: int) -> str:
        """Deallocate a process (NEW)"""
        try:
            if self.memory_manager.deallocate_process(pid):
                return f"✓ Process {pid} deallocated successfully"
            else:
                return f"Process {pid} not found"
        except Exception as e:
            return f"Error deallocating process: {e}"

    def _set_algorithm(self, algorithm: str) -> str:
        """Set page replacement algorithm (NEW)"""
        if self.memory_manager.set_algorithm(algorithm):
            return f"✓ Page replacement algorithm set to {algorithm.upper()}"
        else:
            return f"Invalid algorithm: {algorithm}. Use 'fifo' or 'lru'"

    def _test_memory(self, pattern: str) -> str:
        """Test memory with different access patterns (NEW)"""
        pid = self.memory_manager.create_process("TestProcess", 6)

        result = f"✓ Testing {pattern} memory access pattern...\n"

        if pattern == "sequential":
            for i in range(8):
                page_num = i % 6
                success, message = self.memory_manager.allocate_page(pid, page_num)
                result += f"Access {i+1}: {message}\n"
        elif pattern == "random":
            import random
            for i in range(8):
                page_num = random.randint(0, 5)
                success, message = self.memory_manager.allocate_page(pid, page_num)
                result += f"Access {i+1}: {message}\n"

        return result + "\n" + self._memory_status()

    def _sync_status(self) -> str:
        """Get synchronization status (NEW)"""
        status = self.synchronizer.get_status()

        result = f"""=== PROCESS SYNCHRONIZATION STATUS (NEW) ===
✓ Mutexes: {status['mutexes']}
✓ Semaphores: {status['semaphores']}
✓ Lock Acquisitions: {status['lock_acquisitions']}
✓ Lock Waits: {status['lock_waits']}
Shared Resources: {status['shared_resources']}"""

        # Producer-Consumer status (NEW)
        if self.producer_consumer:
            pc_status = self.producer_consumer.get_status()
            result += f"""

=== PRODUCER-CONSUMER STATUS (NEW) ===
Running: {pc_status['running']}
Buffer: {pc_status['current_buffer']}/{pc_status['buffer_size']}
✓ Items Produced: {pc_status['items_produced']}
✓ Items Consumed: {pc_status['items_consumed']}
Active Producers: {pc_status['active_producers']}
Active Consumers: {pc_status['active_consumers']}"""

        # Dining Philosophers status (NEW)
        if self.dining_philosophers:
            dp_status = self.dining_philosophers.get_status()
            result += f"""

=== DINING PHILOSOPHERS STATUS (NEW) ===
Running: {dp_status['running']}
Philosophers: {dp_status['num_philosophers']}
✓ Total Meals: {dp_status['total_meals']}
✓ Average Meals: {dp_status['avg_meals']:.1f}
✓ Deadlock Preventions: {dp_status['deadlock_preventions']}
States: {', '.join(dp_status['states'])}"""

        return result

    def _handle_mutex(self, operation: str, args: List[str]) -> str:
        """Handle mutex operations (NEW)"""
        if operation == "create" and args:
            name = args[0]
            if self.synchronizer.create_mutex(name):
                return f"✓ Mutex '{name}' created"
            else:
                return f"Mutex '{name}' already exists"
        elif operation == "acquire" and args:
            name = args[0]
            timeout = float(args[1]) if len(args) > 1 else None
            if self.synchronizer.acquire_mutex(name, timeout):
                return f"✓ Mutex '{name}' acquired"
            else:
                return f"Failed to acquire mutex '{name}'"
        elif operation == "release" and args:
            name = args[0]
            if self.synchronizer.release_mutex(name):
                return f"✓ Mutex '{name}' released"
            else:
                return f"Failed to release mutex '{name}'"
        else:
            return "Usage: sync mutex <create|acquire|release> <n> [timeout]"

    def _handle_semaphore(self, operation: str, args: List[str]) -> str:
        """Handle semaphore operations (NEW)"""
        if operation == "create" and args:
            name = args[0]
            value = int(args[1]) if len(args) > 1 else 1
            if self.synchronizer.create_semaphore(name, value):
                return f"✓ Semaphore '{name}' created with value {value}"
            else:
                return f"Semaphore '{name}' already exists"
        elif operation == "acquire" and args:
            name = args[0]
            timeout = float(args[1]) if len(args) > 1 else None
            if self.synchronizer.acquire_semaphore(name, timeout):
                return f"✓ Semaphore '{name}' acquired"
            else:
                return f"Failed to acquire semaphore '{name}'"
        elif operation == "release" and args:
            name = args[0]
            if self.synchronizer.release_semaphore(name):
                return f"✓ Semaphore '{name}' released"
            else:
                return f"Failed to release semaphore '{name}'"
        else:
            return "Usage: sync semaphore <create|acquire|release> <n> [value|timeout]"

    def _handle_producer_consumer(self, args: List[str]) -> str:
        """Handle Producer-Consumer problem (NEW)"""
        if not args:
            return "Usage: sync prodcons <start|stop|status> [producers] [consumers] [duration]"

        operation = args[0].lower()

        if operation == "start":
            if self.producer_consumer and self.producer_consumer.running:
                return "Producer-Consumer already running. Stop it first."

            producers = int(args[1]) if len(args) > 1 else 2
            consumers = int(args[2]) if len(args) > 2 else 2
            duration = int(args[3]) if len(args) > 3 else 10

            self.producer_consumer = ProducerConsumer(buffer_size=5)
            self.producer_consumer.start(producers, consumers, duration)

            return f"✓ Producer-Consumer started: {producers} producers, {consumers} consumers"

        elif operation == "stop":
            if self.producer_consumer:
                self.producer_consumer.stop()
                return "✓ Producer-Consumer stopped"
            else:
                return "No Producer-Consumer running"

        elif operation == "status":
            if self.producer_consumer:
                status = self.producer_consumer.get_status()
                return f"""Producer-Consumer Status:
Running: {status['running']}
Buffer: {status['current_buffer']}/{status['buffer_size']}
✓ Produced: {status['items_produced']}
✓ Consumed: {status['items_consumed']}
Active Producers: {status['active_producers']}
Active Consumers: {status['active_consumers']}
Producer Waits: {status['producer_waits']}
Consumer Waits: {status['consumer_waits']}"""
            else:
                return "No Producer-Consumer instance"
        else:
            return "Usage: sync prodcons <start|stop|status>"

    def _handle_dining_philosophers(self, args: List[str]) -> str:
        """Handle Dining Philosophers problem (NEW)"""
        if not args:
            return "Usage: sync philosophers <start|stop|status> [num_philosophers] [duration]"

        operation = args[0].lower()

        if operation == "start":
            if self.dining_philosophers and self.dining_philosophers.running:
                return "Dining Philosophers already running. Stop it first."

            num_philosophers = int(args[1]) if len(args) > 1 else 5
            duration = int(args[2]) if len(args) > 2 else 15

            self.dining_philosophers = DiningPhilosophers(num_philosophers)

            # Start in background thread
            def start_philosophers():
                self.dining_philosophers.start(duration)

            thread = threading.Thread(target=start_philosophers)
            thread.daemon = True
            thread.start()

            return f"✓ Dining Philosophers started: {num_philosophers} philosophers for {duration}s"

        elif operation == "stop":
            if self.dining_philosophers:
                self.dining_philosophers.stop()
                return "✓ Dining Philosophers stopped"
            else:
                return "No Dining Philosophers running"

        elif operation == "status":
            if self.dining_philosophers:
                status = self.dining_philosophers.get_status()
                meals_str = ", ".join([f"P{i}:{meals}" for i, meals in enumerate(status['meals_eaten'])])
                return f"""Dining Philosophers Status:
Running: {status['running']}
Philosophers: {status['num_philosophers']}
States: {', '.join([f"P{i}:{state}" for i, state in enumerate(status['states'])])}
✓ Total Meals: {status['total_meals']}
✓ Average Meals: {status['avg_meals']:.1f}
Meals: {meals_str}
✓ Deadlock Preventions: {status['deadlock_preventions']}
Active Threads: {status['active_threads']}"""
            else:
                return "No Dining Philosophers instance"
        else:
            return "Usage: sync philosophers <start|stop|status>"

    def get_memory_manager(self):
        """Get memory manager instance (NEW)"""
        return self.memory_manager

    def get_synchronizer(self):
        """Get synchronizer instance (NEW)"""
        return self.synchronizer