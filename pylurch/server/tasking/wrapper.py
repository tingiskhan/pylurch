from typing import Callable, Tuple, Dict, Any
from pylurch.contract import enums as e, interfaces as i, database as db
from logging import Logger
from .task import TaskWrapper
from ...utils import make_base_logger


class BaseWrapper(object):
    def __init__(self, interface: i.DatabaseInterface, logger: Logger = None):
        """
        Base class for enqueuing tasks.
        """

        self._i = interface
        self._logger = logger or make_base_logger(self.__class__.__name__)

    def make_task(self, f, *args, **kwargs) -> TaskWrapper:
        raise NotImplementedError()

    def enqueue(self, f: Callable[[Tuple[Any], Dict[str, Any]], Any], *args, **kwargs) -> str:
        task = self.make_task(f, *args, **kwargs)
        self._enqueue(task)

        return task.key

    def _enqueue(self, task: TaskWrapper):
        raise NotImplementedError()

    def get_task(self, key):
        return self._i.make_interface(db.Task).get(lambda u: u.key == key, one=True)

    def check_status(self, key):
        task = self.get_task(key)

        if task is None:
            return e.Status.Unknown

        return task.status

    def get_result(self, task_id: str) -> Any:
        raise NotImplementedError()
