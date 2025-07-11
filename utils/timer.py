"""
timer.py

A light-weight profiling utility that makes it trivial to time multiple
named sections of code without adding too much boiler-plate.

Core ideas
----------
StatusTimer
    Keeps timing statistics for *one* particular status / code segment:
        • total accumulated time
        • number of completed runs
        • mean time in milliseconds (avgms)

Timer
    A registry that holds a collection of `StatusTimer` objects, one for
    each unique status string.  When `activate()` is called the user can
    simply sprinkle `timer("my_status")` around the code to measure the
    execution time of these blocks.

Example
-------
>>> timer = Timer(activate=True)          # or timer.activate()
>>> timer("io");  ...   timer("io")       # toggles start/stop
>>> timer("model"); ... timer("model")    # another segment

>>> timer.summarize()
status: io,    avg time: 12.4512ms, total time: 0.0498s
status: model, avg time:  1.8734ms, total time: 0.2252s
"""

import time
from typing import Dict, Optional

from pigmento import pnt  # coloured/pretty print helper


# =============================================================================
#                                StatusTimer
# =============================================================================
class StatusTimer:
    """
    Handles the timing of *one* logical code segment that can be toggled
    on/off using `run()`.  The method is intentionally idempotent:

        timer.run()   # start
        ...
        timer.run()   # stop & accumulate

    Subsequent pairs accumulate more data allowing us to compute the mean
    execution time for that segment.
    """

    def __init__(self, total_count=0):
        self.total_time: float = 0.0     # seconds accumulated so far
        self.start_time: Optional[float] = None
        self.timing: bool = False        # are we currently measuring?
        self.count: int = 0              # how many completed cycles
        self.total_count = total_count

    # ---------------------------------------------------------------------
    # Toggle timing
    # ---------------------------------------------------------------------
    def run(self):
        crt_time = time.time()
        if self.timing:
            # We are *stopping* the timer: accumulate and switch off
            self.total_time += crt_time - self.start_time
            self.timing = False
            self.count += 1

            if self.total_count and self.count >= self.total_count:
                raise StopIteration
        else:
            # We are *starting* the timer
            self.timing = True
            self.start_time = crt_time

    # ---------------------------------------------------------------------
    # Reset statistics
    # ---------------------------------------------------------------------
    def clear(self):
        self.total_time = 0.0
        self.start_time = None
        self.timing = False
        self.count = 0

    # ---------------------------------------------------------------------
    # Mean time in milliseconds
    # ---------------------------------------------------------------------
    def avgms(self):
        if not self.count:
            return "unavailable"
        return self.total_time * 1000 / self.count


# =============================================================================
#                                   Timer
# =============================================================================
class Timer:
    """
    A *manager* class that coordinates multiple StatusTimers.

    Usage pattern
    -------------
    >>> timer = Timer(activate=True)            # or activate later
    >>> timer("db");  ...  timer("db")          # time DB section
    >>> timer("compute"); ... timer("compute")  # time compute section
    >>> timer.summarize()                       # print report
    """

    def __init__(self, activate: bool = False):
        self.status_dict: Dict[str, StatusTimer] = {}
        self.total_counts = dict()
        self._activate = activate

    # ---------------------------------------------------------------------
    # Control switches
    # ---------------------------------------------------------------------
    def activate(self):
        """Enable timing globally."""
        self._activate = True

    def deactivate(self):
        """Disable timing globally (no-ops thereafter)."""
        self._activate = False

    def set_total_count(self, status, count):
        self.total_counts[status] = count

    # ---------------------------------------------------------------------
    # Start / stop timing for a given status
    # ---------------------------------------------------------------------

    def run(self, status: str):
        if not self._activate:
            return
        if status not in self.status_dict:
            self.status_dict[status] = StatusTimer(total_count=self.total_counts.get(status, 0))
        self.status_dict[status].run()

    # Allow `Timer` object to be called like a function -------------------
    def __call__(self, status: str):
        return self.run(status)

    # ---------------------------------------------------------------------
    # Reset all statuses
    # ---------------------------------------------------------------------
    def clear(self):
        for status in self.status_dict.values():
            status.clear()

    # ---------------------------------------------------------------------
    # Pretty print summary
    # ---------------------------------------------------------------------
    def summarize(self):
        for status, stat in self.status_dict.items():
            pnt(
                f"status: {status}, "
                f"avg time: {stat.avgms():.4f}ms, "
                f"total time: {stat.total_time:.4f}s"
            )
