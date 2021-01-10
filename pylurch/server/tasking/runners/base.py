from typing import Callable, Any, Optional
from logging import Logger
from ..tasks import BaseTask
from pyalfred.server.utils import make_base_logger
from pylurch.contract import enums as e, database as db
from pyalfred.contract.client import Client


class BaseRunner(object):
    def __init__(self, client: Client, logger: Logger = None):
        """
        Base class for enqueuing tasks.
        """

        self._client = client
        self._logger = logger or make_base_logger(self.__class__.__name__)

    def make_task(self, f, *args, **kwargs) -> BaseTask:
        raise NotImplementedError()

    def enqueue(self, f: Callable[..., Any], *args, **kwargs) -> str:
        task = self.make_task(f, *args, **kwargs)
        self._enqueue(task)

        return task.key

    def _enqueue(self, task: BaseTask):
        raise NotImplementedError()

    def get_task(self, key) -> db.Task:
        return self._client.get(db.Task, lambda u: u.key == key, one=True)

    def check_status(self, key):
        task = self.get_task(key)

        if task is None:
            return e.Status.Unknown

        return task.status

    def get_result(self, task_id: str) -> Any:
        raise NotImplementedError()

    def get_exception(self, key: str) -> Optional[db.TaskException]:
        task = self._client.get(db.Task, lambda u: u.key == key, one=True)

        return self._client.get(db.TaskException, lambda u: u.task_id == task.id, one=True)
