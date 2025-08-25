import os
import platform
import multiprocessing as mp
from multiprocessing.context import BaseContext
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice
from typing import Callable, Iterable, Iterator, Any, Dict, List, Tuple, Optional


# ---------- private helper ----------
def _run_chunk(func: Callable[[Any], Any], idx_items: List[Tuple[int, Any]]) -> List[Tuple[int, Any]]:
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

    def __init__(self, processes: Optional[int] = None, use_threads: bool = True):
        """
        Initialize UniversalPool with optimal worker selection.

        Args:
            processes: Number of workers (defaults to CPU count)
            use_threads: Use threads instead of processes (default True).
                        Threads avoid serialization overhead and work well
                        when the GIL is released (NumPy, I/O operations).
        """
        self.processes = processes or os.cpu_count()
        self._use_threads = use_threads
        self._is_native_pool = False
        # _pool can be a multiprocessing.Pool or an Executor depending on platform/settings
        self._pool: Any
        # reusable context that may be None on platforms without get_context
        ctx: Optional[BaseContext] = None

        if use_threads:
            self._pool = ThreadPoolExecutor(max_workers=self.processes)
        else:
            # Try native multiprocessing.Pool first (most efficient)
            try:
                if hasattr(mp, "get_context") and platform.system() not in ("Windows", "Darwin"):
                    ctx = mp.get_context("fork")
                    self._pool = ctx.Pool(processes=self.processes)
                    self._is_native_pool = True
                else:
                    raise ValueError("Use ProcessPoolExecutor fallback")
            except (ValueError, AttributeError):
                # Fall back to ProcessPoolExecutor with robust context selection
                try:
                    if hasattr(mp, "get_context") and platform.system() not in ("Windows", "Darwin"):
                        ctx = mp.get_context("fork")
                        self._pool = ProcessPoolExecutor(max_workers=self.processes, mp_context=ctx)
                    else:
                        # Fall back to spawn (Windows, macOS, or when fork unavailable)
                        ctx = mp.get_context("spawn") if hasattr(mp, "get_context") else None
                        if ctx:
                            self._pool = ProcessPoolExecutor(max_workers=self.processes, mp_context=ctx)
                        else:
                            # Python 3.8 fallback
                            self._pool = ProcessPoolExecutor(max_workers=self.processes)
                except (ValueError, AttributeError):
                    # Final fallback to basic ProcessPoolExecutor
                    self._pool = ProcessPoolExecutor(max_workers=self.processes)

    # ---------- public API ----------
    def imap(
        self, func: Callable[[Any], Any], iterable: Iterable[Any], chunksize: int = 1, max_prefetch: int = 4
    ) -> Iterator[Any]:
        """
        Stream results *in order* like multiprocessing.Pool.imap.
        `max_prefetch` limits outstanding chunks to bound RAM (ignored for native pools).
        """
        if self._is_native_pool:
            # Use native imap (ignores max_prefetch but is very efficient)
            return self._pool.imap(func, iterable, chunksize)
        else:
            # Use streaming implementation for ProcessPoolExecutor/ThreadPoolExecutor
            return self._streaming_imap(func, iterable, chunksize, max_prefetch)

    def _streaming_imap(
        self, func: Callable[[Any], Any], iterable: Iterable[Any], chunksize: int = 1, max_prefetch: int = 4
    ) -> Iterator[Any]:
        """Streaming imap implementation for ExecutorPool-based backends."""
        # producer over the input
        it = enumerate(iterable)  # keeps original positions
        next_yield = 0  # next index expected to yield
        buffer: Dict[int, Any] = {}  # completed results waiting to be yielded
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
        if self._is_native_pool:
            self._pool.close()
            self._pool.join()
        else:
            self._pool.shutdown(wait=True)


# convenience factory
def get_pool(processes: Optional[int] = None, threads: bool = True) -> UniversalPool:
    """Get a pool that works on all platforms with optimal performance."""
    return UniversalPool(processes, use_threads=threads)
