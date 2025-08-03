#!/usr/bin/env python3
"""
memory_manager.py - Memory Management Module for Advanced Shell - Deliverable 3
Implements paging system with FIFO and LRU page replacement algorithms
✓ Paging system with fixed-size page frames (NEW)
✓ FIFO and LRU page replacement algorithms (NEW)
✓ Page fault handling and tracking (NEW)
✓ Memory overflow simulation (NEW)
"""

import threading
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import OrderedDict


class PageReplacementAlgorithm(Enum):
    FIFO = "fifo"
    LRU = "lru"


class Page:
    """Represents a memory page (NEW)"""

    def __init__(self, page_id: int, process_id: int):
        self.page_id = page_id
        self.process_id = process_id
        self.last_accessed = time.time()
        self.loaded_time = time.time()

    def access(self):
        """Mark page as accessed for LRU tracking (NEW)"""
        self.last_accessed = time.time()

    def __str__(self):
        return f"P{self.process_id}:Pg{self.page_id}"


class Process:
    """Process with memory requirements (NEW)"""

    def __init__(self, pid: int, name: str, pages_needed: int):
        self.pid = pid
        self.name = name
        self.pages_needed = pages_needed
        self.page_table: Dict[int, Optional[Page]] = {
            i: None for i in range(pages_needed)}


class MemoryManager:
    """Memory Manager with paging and page replacement (NEW - Deliverable 3)"""

    def __init__(self, total_frames: int = 8):
        # Physical memory simulation
        self.total_frames = total_frames
        self.physical_memory: List[Optional[Page]] = [None] * total_frames
        self.free_frames: List[int] = list(range(total_frames))
        self.used_frames: Dict[int, Page] = {}

        # Page replacement algorithm
        self.replacement_algorithm = PageReplacementAlgorithm.FIFO

        # Statistics (NEW)
        self.page_faults = 0
        self.page_hits = 0
        self.page_replacements = 0

        # Process management
        self.processes: Dict[int, Process] = {}
        self.next_pid = 1
        self.lock = threading.RLock()

        # Algorithm-specific data structures (NEW)
        self.fifo_queue: List[int] = []  # For FIFO replacement
        self.lru_access_order = OrderedDict()  # For LRU replacement

    def create_process(self, name: str, pages_needed: int) -> int:
        """Create a new process with memory requirements (NEW)"""
        with self.lock:
            pid = self.next_pid
            self.next_pid += 1
            self.processes[pid] = Process(pid, name, pages_needed)
            return pid

    def allocate_page(self, pid: int, page_number: int) -> Tuple[bool, str]:
        """Allocate a page for a process (NEW)"""
        with self.lock:
            if pid not in self.processes:
                return False, f"Process {pid} not found"

            process = self.processes[pid]
            if page_number >= process.pages_needed:
                return False, f"Invalid page number {page_number}"

            # Check if page already allocated (page hit)
            if process.page_table[page_number] is not None:
                self.page_hits += 1
                page = process.page_table[page_number]
                page.access()
                self._update_lru_access(page)
                return True, f"✓ Page hit: {page}"

            # Page fault occurred (NEW)
            self.page_faults += 1

            page = Page(page_number, pid)
            frame_number = self._allocate_frame(page)

            if frame_number == -1:
                return False, "No memory available"

            process.page_table[page_number] = page
            return True, f"✓ Page allocated: {page} -> Frame {frame_number}"

    def _allocate_frame(self, page: Page) -> int:
        """Allocate a physical frame (NEW)"""
        # Use free frame if available
        if self.free_frames:
            frame_number = self.free_frames.pop(0)
            self.physical_memory[frame_number] = page
            self.used_frames[frame_number] = page
            self.fifo_queue.append(frame_number)
            self._update_lru_access(page)
            return frame_number

        # Need page replacement (NEW)
        return self._replace_page(page)

    def _replace_page(self, new_page: Page) -> int:
        """Replace a page using selected algorithm (NEW)"""
        self.page_replacements += 1

        if self.replacement_algorithm == PageReplacementAlgorithm.FIFO:
            return self._fifo_replace(new_page)
        else:  # LRU
            return self._lru_replace(new_page)

    def _fifo_replace(self, new_page: Page) -> int:
        """FIFO page replacement algorithm (NEW)"""
        frame_to_replace = self.fifo_queue.pop(0)
        old_page = self.physical_memory[frame_to_replace]

        if old_page:
            # Update process page table
            if old_page.process_id in self.processes:
                self.processes[old_page.process_id].page_table[old_page.page_id] = None

        self.physical_memory[frame_to_replace] = new_page
        self.used_frames[frame_to_replace] = new_page
        self.fifo_queue.append(frame_to_replace)
        self._update_lru_access(new_page)

        return frame_to_replace

    def _lru_replace(self, new_page: Page) -> int:
        """LRU page replacement algorithm (NEW)"""
        # Find least recently used page
        lru_page = None
        lru_frame = -1

        for frame_num, page in self.used_frames.items():
            if lru_page is None or page.last_accessed < lru_page.last_accessed:
                lru_page = page
                lru_frame = frame_num

        if lru_page and lru_page.process_id in self.processes:
            self.processes[lru_page.process_id].page_table[lru_page.page_id] = None

        self.physical_memory[lru_frame] = new_page
        self.used_frames[lru_frame] = new_page
        self._update_lru_access(new_page)

        return lru_frame

    def _update_lru_access(self, page: Page):
        """Update LRU tracking (NEW)"""
        page_key = f"{page.process_id}:{page.page_id}"
        if page_key in self.lru_access_order:
            del self.lru_access_order[page_key]
        self.lru_access_order[page_key] = page

    def set_algorithm(self, algorithm: str) -> bool:
        """Set page replacement algorithm (NEW)"""
        if algorithm.lower() == "fifo":
            self.replacement_algorithm = PageReplacementAlgorithm.FIFO
            return True
        elif algorithm.lower() == "lru":
            self.replacement_algorithm = PageReplacementAlgorithm.LRU
            return True
        return False

    def get_status(self) -> Dict:
        """Get memory status (NEW)"""
        with self.lock:
            used_frames = len(self.used_frames)
            free_frames = len(self.free_frames)

            return {
                "total_frames": self.total_frames,
                "used_frames": used_frames,
                "free_frames": free_frames,
                "utilization": (used_frames / self.total_frames) * 100,
                "page_faults": self.page_faults,
                "page_hits": self.page_hits,
                "replacements": self.page_replacements,
                "hit_ratio": (self.page_hits / max(1, self.page_hits + self.page_faults)) * 100,
                "algorithm": self.replacement_algorithm.value,
                "processes": len(self.processes)
            }

    def reset_memory(self) -> bool:
        """Reset all memory management state and statistics (NEW)"""
        with self.lock:
            # Reset physical memory
            self.physical_memory = [None] * self.total_frames
            self.free_frames = list(range(self.total_frames))
            self.used_frames.clear()

            # Reset statistics
            self.page_faults = 0
            self.page_hits = 0
            self.page_replacements = 0

            # Reset processes
            self.processes.clear()
            self.next_pid = 1

            # Reset algorithm data structures
            self.fifo_queue.clear()
            self.lru_access_order.clear()

            return True

    def deallocate_process(self, pid: int) -> bool:
        """Deallocate all pages for a process (NEW)"""
        with self.lock:
            if pid not in self.processes:
                return False

            process = self.processes[pid]
            freed_frames = 0

            # Free all allocated pages
            for page_num, page in process.page_table.items():
                if page is not None:
                    for frame_num, frame_page in self.used_frames.items():
                        if frame_page == page:
                            self.physical_memory[frame_num] = None
                            del self.used_frames[frame_num]
                            self.free_frames.append(frame_num)
                            if frame_num in self.fifo_queue:
                                self.fifo_queue.remove(frame_num)
                            freed_frames += 1
                            break

            del self.processes[pid]
            return True

    def simulate_memory_access(self, pid: int, pattern: str = "random", count: int = 8):
        """Simulate memory access patterns for testing (NEW)"""
        if pid not in self.processes:
            return

        process = self.processes[pid]

        import random
        for i in range(count):
            if pattern == "sequential":
                page_num = i % process.pages_needed
            else:  # random
                page_num = random.randint(0, process.pages_needed - 1)

            success, message = self.allocate_page(pid, page_num)
            time.sleep(0.1)