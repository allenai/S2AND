import multiprocessing as mp
import os
import platform
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from itertools import islice
from typing import Any


# ---------- private helper ----------
def _run_chunk(func: Callable[[Any], Any], idx_items: list[tuple[int, Any]]) -> list[tuple[int, Any]]:
    """Run func on each item in the chunk and keep the index."""
    out = []
    for idx, item in idx_items:
        try:
            out.append((idx, func(item)))
        except Exception as e:  # propagate which element failed
            raise RuntimeError(f"imap item {idx} raised") from e
    return out


# ---------- main class ----------
class UniversalPool:
    """
    Almost-drop-in replacement for multiprocessing.Pool on Py 3.11+
    with ordered streaming imap and cross-platform support.
    """

    def __init__(self, processes: int | None = None, use_threads: bool = True):
        """
        Initialize UniversalPool with optimal worker selection.

        Args:
            processes: Number of workers (defaults to CPU count)
            use_threads: Use threads instead of processes (default True).
                        Threads avoid serialization overhead and work well
                        when the GIL is released (NumPy, I/O operations).
        """
        self.processes = processes or os.cpu_count()
        self._pool: ProcessPoolExecutor | ThreadPoolExecutor

        if use_threads:
            self._pool = ThreadPoolExecutor(max_workers=self.processes)
        else:
            # fork on Linux (fast, avoids re-import); spawn on Windows/macOS (required)
            if platform.system() not in ("Windows", "Darwin"):
                ctx = mp.get_context("fork")
            else:
                ctx = mp.get_context("spawn")
            self._pool = ProcessPoolExecutor(max_workers=self.processes, mp_context=ctx)

    # ---------- public API ----------
    def imap(
        self, func: Callable[[Any], Any], iterable: Iterable[Any], chunksize: int = 1, max_prefetch: int = 4
    ) -> Iterator[Any]:
        """
        Stream results *in order* like multiprocessing.Pool.imap.
        `max_prefetch` limits outstanding chunks to bound RAM.
        """
        return self._streaming_imap(func, iterable, chunksize, max_prefetch)

    def _streaming_imap(
        self, func: Callable[[Any], Any], iterable: Iterable[Any], chunksize: int = 1, max_prefetch: int = 4
    ) -> Iterator[Any]:
        """Streaming imap implementation with backpressure control."""
        # producer over the input
        it = enumerate(iterable)  # keeps original positions
        next_yield = 0  # next index expected to yield
        buffer: dict[int, Any] = {}  # completed results waiting to be yielded
        pending = set()

        def submit_chunk():
            chunk = list(islice(it, chunksize))
            if chunk:
                fut = self._pool.submit(_run_chunk, func, chunk)
                pending.add(fut)
                return True
            return False

        # prime the pipeline
        for _ in range(max_prefetch):
            if not submit_chunk():
                break

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                pending.remove(fut)
                for idx, res in fut.result():
                    buffer[idx] = res
                # keep queue topped-up
                submit_chunk()

            # yield any ready-in-order items
            while next_yield in buffer:
                yield buffer.pop(next_yield)
                next_yield += 1

    # ---------- context manager ----------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._pool.shutdown(wait=True)


# convenience factory
def get_pool(processes: int | None = None, threads: bool = True) -> UniversalPool:
    """Get a pool that works on all platforms with optimal performance."""
    return UniversalPool(processes, use_threads=threads)
