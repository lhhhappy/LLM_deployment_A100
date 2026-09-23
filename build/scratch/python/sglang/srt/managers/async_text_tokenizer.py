"""Serialize full tokenizations off the HTTP event loop (no prefix cache)."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial


class AsyncTextTokenizer:
    """One worker per manager; canceled clients cannot build an executor backlog.

    HF wrappers can mutate padding/truncation state even for encode calls. Keep
    all regular text calls, including short ones, on the same serial worker.
    """

    def __init__(self):
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sglang-tokenize"
        )
        self._slot = asyncio.Semaphore(1)
        self._closed = False

    async def run(self, fn, *args):
        await self._slot.acquire()
        try:
            if self._closed:
                raise RuntimeError("text tokenizer is closed")
            future = asyncio.get_running_loop().run_in_executor(
                self._executor, partial(fn, *args)
            )
        except BaseException:
            self._slot.release()
            raise

        def completed(future):
            # A canceled caller does not stop Rust work. Release only when that
            # work really ends, and consume abandoned exceptions on the loop.
            self._slot.release()
            if not future.cancelled():
                future.exception()

        future.add_done_callback(completed)
        return await asyncio.shield(future)

    def close(self):
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
