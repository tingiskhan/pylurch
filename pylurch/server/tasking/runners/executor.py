from concurrent.futures import Executor, ThreadPoolExecutor, Future
from cachetools import TTLCache
from ..tasks import BaseTask
from .base import BaseRunner
from pylurch.contract import enums as e, database as db


class ExecutorRunner(BaseRunner):
    def __init__(self, client, executor: Executor = None):
        """
        Class for enqueuing tasks using 'concurrent.futures.Executor' as task manager. Do note that this is for
        debugging purposes rather than production use.
        """

        super().__init__(client)

        self._exc = executor or ThreadPoolExecutor()
        self._results = TTLCache(maxsize=1_000_000_000, ttl=60 * 60)

    def _enqueue(self, task):
        task.initialize()
        future = self._exc.submit(task._f, *task._args, **task._kwargs)
        future.add_done_callback(lambda u: self._done_callback(u, task_id=task.db.id, key=task.key))

    def _done_callback(self, u: Future, task_id: int, key: str):
        task = self._client.get(db.Task, lambda x: x.id == task_id, one=True)

        if u.done():
            self._results.update({key: u.result()})
            task.status = e.Status.Done
        else:
            task.status = e.Status.Failed

        self._client.update(task)

    def make_task(self, f, *args, **kwargs) -> BaseTask:
        return BaseTask(f, self._client, args=args, kwargs=kwargs)

    def get_result(self, task_id):
        return self._results.get(task_id, None)
