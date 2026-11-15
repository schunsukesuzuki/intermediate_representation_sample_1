"""One active four-method comparison per API process; async HTTP polling.

In-memory task state: restart loses progress, but last completed PDF remains on disk.
Do not run multiple uvicorn workers with this in-memory manager.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Callable
from uuid import uuid4


class ComparisonJobs:
    def __init__(self) -> None:
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gnn-compare')
        self._jobs: dict[str, dict] = {}

    def submit(self, fn: Callable[[Callable[[str], None]], dict]) -> str:
        with self._lock:
            if any(j['status'] in ('queued', 'running') for j in self._jobs.values()):
                raise RuntimeError('A four-way comparison is already running. Poll its status before starting another.')
            # Avoid unbounded retention when the demo is left running for days.
            for key in list(self._jobs)[:-4]:
                if self._jobs[key]['status'] not in ('queued', 'running'):
                    self._jobs.pop(key, None)
            job_id = uuid4().hex
            self._jobs[job_id] = {'job_id': job_id, 'status': 'queued', 'stage': 'queued',
                                  'created_utc': datetime.now(timezone.utc).isoformat()}
        def update(stage: str) -> None:
            with self._lock:
                entry = self._jobs[job_id]
                entry.update(status='running', stage=stage)
        def worker() -> None:
            try:
                update('initializing')
                result = fn(update)
                with self._lock:
                    self._jobs[job_id].update(status='completed', stage='done', result=result)
            except Exception as exc:
                with self._lock:
                    self._jobs[job_id].update(status='failed', stage='failed',
                                              error=f'{type(exc).__name__}: {exc}')
        self._executor.submit(worker)
        return job_id

    def status(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None
