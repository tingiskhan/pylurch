from .task import TaskWrapper
from .wrapper import BaseWrapper
from concurrent.futures import Executor, ThreadPoolExecutor, Future
from cachetools import TTLCache


class ExecutorWrapper(BaseWrapper):
    def __init__(self, interface, executor: Executor = None):
        """
        Class for enqueuing tasks using 'concurrent.futures.Executor' as task manager. Do note that this is for
        debugging purposes rather than production use.
        """

        super().__init__(interface)

        self._exc = executor or ThreadPoolExecutor()
        self._results = TTLCache(maxsize=1000, ttl=500)

    def _enqueue(self, task):
        task.initialize()
        future = self._exc.submit(task._f, *task._args, **task._kwargs)
        future.add_done_callback(lambda u: self._done_callback(u, key=task.key))

    def _done_callback(self, u: Future, key: str):
        self._results[key] = u.result()

    def make_task(self, f, *args, **kwargs) -> TaskWrapper:
        return TaskWrapper(f, self._i, args=args, kwargs=kwargs)

    def get_result(self, task_id):
        return self._results.get(task_id, None)
