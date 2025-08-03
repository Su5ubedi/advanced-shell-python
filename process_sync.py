#!/usr/bin/env python3
"""
process_sync.py - Process Synchronization Module for Advanced Shell - Deliverable 3
Implements mutexes, semaphores, and classical synchronization problems
✓ Mutexes and semaphores for resource synchronization (NEW)
✓ Producer-Consumer problem implementation (NEW)
✓ Dining Philosophers problem implementation (NEW)
✓ Race condition prevention and deadlock avoidance (NEW)
"""

import threading
import time
import random
import queue
from typing import Dict, List
from enum import Enum


class PhilosopherState(Enum):
    THINKING = "thinking"
    HUNGRY = "hungry"
    EATING = "eating"


class ProcessSynchronizer:
    """Main synchronization manager (NEW - Deliverable 3)"""

    def __init__(self):
        self.mutexes: Dict[str, threading.Lock] = {}
        self.semaphores: Dict[str, threading.Semaphore] = {}
        self.shared_resources: Dict[str, any] = {}

        # Statistics (NEW)
        self.lock_acquisitions = 0
        self.lock_waits = 0

        self.main_lock = threading.RLock()

    def create_mutex(self, name: str) -> bool:
        """Create a named mutex (NEW)"""
        with self.main_lock:
            if name in self.mutexes:
                return False
            self.mutexes[name] = threading.Lock()
            return True

    def create_semaphore(self, name: str, initial_value: int = 1) -> bool:
        """Create a named semaphore (NEW)"""
        with self.main_lock:
            if name in self.semaphores:
                return False
            self.semaphores[name] = threading.Semaphore(initial_value)
            return True

    def acquire_mutex(self, name: str, timeout: float = None) -> bool:
        """Acquire a mutex (NEW)"""
        if name not in self.mutexes:
            return False

        self.lock_acquisitions += 1
        try:
            if timeout is None:
                acquired = self.mutexes[name].acquire(blocking=True)
            else:
                acquired = self.mutexes[name].acquire(timeout=timeout)
            if not acquired:
                self.lock_waits += 1
            return acquired
        except Exception:
            self.lock_waits += 1
            return False

    def release_mutex(self, name: str) -> bool:
        """Release a mutex (NEW)"""
        if name not in self.mutexes:
            return False
        try:
            self.mutexes[name].release()
            return True
        except:
            return False

    def acquire_semaphore(self, name: str, timeout: float = None) -> bool:
        """Acquire a semaphore (NEW)"""
        if name not in self.semaphores:
            return False
        try:
            if timeout is None:
                return self.semaphores[name].acquire(blocking=True)
            else:
                return self.semaphores[name].acquire(timeout=timeout)
        except Exception:
            return False

    def release_semaphore(self, name: str) -> bool:
        """Release a semaphore (NEW)"""
        if name not in self.semaphores:
            return False
        try:
            self.semaphores[name].release()
            return True
        except:
            return False

    def get_status(self) -> Dict:
        """Get synchronization status (NEW)"""
        return {
            "mutexes": len(self.mutexes),
            "semaphores": len(self.semaphores),
            "lock_acquisitions": self.lock_acquisitions,
            "lock_waits": self.lock_waits,
            "shared_resources": len(self.shared_resources)
        }


class ProducerConsumer:
    """Producer-Consumer synchronization problem (NEW - Deliverable 3)"""

    def __init__(self, buffer_size: int = 5):
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
        self.running = False
        self.producers: List[threading.Thread] = []
        self.consumers: List[threading.Thread] = []

        # Synchronization primitives (NEW)
        self.mutex = threading.Lock()  # Protects buffer access
        self.not_full = threading.Semaphore(buffer_size)   # Producers wait when full
        self.not_empty = threading.Semaphore(0)            # Consumers wait when empty

        # Statistics (NEW)
        self.items_produced = 0
        self.items_consumed = 0
        self.producer_waits = 0
        self.consumer_waits = 0

    def producer_task(self, producer_id: int, items_to_produce: int = 10):
        """Producer thread function (NEW)"""
        for i in range(items_to_produce):
            if not self.running:
                break

            item = f"Item-P{producer_id}-{i}"

            # Wait if buffer is full
            if not self.not_full.acquire(timeout=1):
                self.producer_waits += 1
                continue

            try:
                with self.mutex:
                    if self.running:
                        self.buffer.put(item, block=False)
                        self.items_produced += 1

                self.not_empty.release()  # Signal consumers
                time.sleep(random.uniform(0.1, 0.5))

            except queue.Full:
                self.not_full.release()

    def consumer_task(self, consumer_id: int, items_to_consume: int = 10):
        """Consumer thread function (NEW)"""
        for i in range(items_to_consume):
            if not self.running:
                break

            # Wait if buffer is empty
            if not self.not_empty.acquire(timeout=1):
                self.consumer_waits += 1
                continue

            try:
                with self.mutex:
                    if self.running and not self.buffer.empty():
                        item = self.buffer.get(block=False)
                        self.items_consumed += 1

                self.not_full.release()  # Signal producers
                time.sleep(random.uniform(0.1, 0.3))

            except queue.Empty:
                self.not_empty.release()

    def start(self, num_producers: int = 2, num_consumers: int = 2, duration: int = 10):
        """Start the producer-consumer simulation (NEW)"""
        self.running = True
        self.items_produced = 0
        self.items_consumed = 0

        # Create producer threads
        for i in range(num_producers):
            thread = threading.Thread(target=self.producer_task, args=(i, duration))
            self.producers.append(thread)
            thread.start()

        # Create consumer threads
        for i in range(num_consumers):
            thread = threading.Thread(target=self.consumer_task, args=(i, duration))
            self.consumers.append(thread)
            thread.start()

    def stop(self):
        """Stop the simulation (NEW)"""
        self.running = False

        for thread in self.producers + self.consumers:
            thread.join(timeout=1)

        self.producers.clear()
        self.consumers.clear()

    def get_status(self) -> Dict:
        """Get simulation status (NEW)"""
        # Check if all threads have finished
        active_producers = len([t for t in self.producers if t.is_alive()])
        active_consumers = len([t for t in self.consumers if t.is_alive()])

        # Auto-update running status if no active threads
        if self.running and active_producers == 0 and active_consumers == 0 and len(self.producers) > 0:
            self.running = False

        return {
            "buffer_size": self.buffer_size,
            "current_buffer": self.buffer.qsize(),
            "items_produced": self.items_produced,
            "items_consumed": self.items_consumed,
            "producer_waits": self.producer_waits,
            "consumer_waits": self.consumer_waits,
            "active_producers": active_producers,
            "active_consumers": active_consumers,
            "running": self.running
        }


class DiningPhilosophers:
    """Dining Philosophers synchronization problem (NEW - Deliverable 3)"""

    def __init__(self, num_philosophers: int = 5):
        self.num_philosophers = num_philosophers
        self.philosophers: List[threading.Thread] = []
        self.forks = [threading.Lock() for _ in range(num_philosophers)]
        self.states = [PhilosopherState.THINKING for _ in range(num_philosophers)]
        self.running = False

        # Statistics (NEW)
        self.meals_eaten = [0] * num_philosophers
        self.deadlock_prevention_count = 0

        self.state_lock = threading.Lock()

    def philosopher_task(self, philosopher_id: int):
        """Philosopher thread function with deadlock prevention (NEW)"""
        left_fork = philosopher_id
        right_fork = (philosopher_id + 1) % self.num_philosophers

        while self.running:
            # Think
            self._set_state(philosopher_id, PhilosopherState.THINKING)
            time.sleep(random.uniform(1, 3))

            # Get hungry
            self._set_state(philosopher_id, PhilosopherState.HUNGRY)

            # Deadlock prevention: even philosophers pick left first, odd pick right first
            if philosopher_id % 2 == 0:
                first_fork, second_fork = left_fork, right_fork
            else:
                first_fork, second_fork = right_fork, left_fork
                self.deadlock_prevention_count += 1

            # Try to acquire forks
            if self.forks[first_fork].acquire(timeout=2):
                if self.forks[second_fork].acquire(timeout=2):
                    # Eat
                    self._set_state(philosopher_id, PhilosopherState.EATING)
                    self.meals_eaten[philosopher_id] += 1
                    time.sleep(random.uniform(1, 2))

                    # Release forks
                    self.forks[second_fork].release()
                    self.forks[first_fork].release()
                else:
                    # Couldn't get second fork
                    self.forks[first_fork].release()

    def _set_state(self, philosopher_id: int, state: PhilosopherState):
        """Thread-safe state update (NEW)"""
        with self.state_lock:
            self.states[philosopher_id] = state

    def start(self, duration: int = 20):
        """Start the dining philosophers simulation (NEW)"""
        self.running = True

        # Reset statistics
        self.meals_eaten = [0] * self.num_philosophers
        self.deadlock_prevention_count = 0

        # Create philosopher threads
        for i in range(self.num_philosophers):
            thread = threading.Thread(target=self.philosopher_task, args=(i,))
            self.philosophers.append(thread)
            thread.start()

        # Run for specified duration
        time.sleep(duration)
        self.stop()

    def stop(self):
        """Stop the simulation (NEW)"""
        self.running = False

        for thread in self.philosophers:
            thread.join(timeout=1)

        self.philosophers.clear()

    def get_status(self) -> Dict:
        """Get simulation status (NEW)"""
        with self.state_lock:
            return {
                "num_philosophers": self.num_philosophers,
                "states": [state.value for state in self.states],
                "meals_eaten": self.meals_eaten.copy(),
                "total_meals": sum(self.meals_eaten),
                "avg_meals": sum(self.meals_eaten) / self.num_philosophers,
                "deadlock_preventions": self.deadlock_prevention_count,
                "running": self.running,
                "active_threads": len([t for t in self.philosophers if t.is_alive()])
            }