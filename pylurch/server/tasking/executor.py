from .task import TaskWrapper
from .wrapper import BaseWrapper
from concurrent.futures import Executor, ThreadPoolExecutor


class ExecutorWrapper(BaseWrapper):
    def __init__(self, interface, executor: Executor = None):
        """
        Class for enqueuing tasks using 'concurrent.futures.Executor' as task manager.
        """

        super().__init__(interface)

        self._exc = executor or ThreadPoolExecutor()

    def _enqueue(self, task):
        task.initialize()
        self._exc.submit(task._f, *task._args, **task._kwargs)

    def make_task(self, f, *args, **kwargs) -> TaskWrapper:
        return TaskWrapper(f, self._i, args=args, kwargs=kwargs)