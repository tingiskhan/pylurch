from typing import Dict, Callable, Tuple, Any
from datetime import datetime
from uuid import uuid4
from pylurch.contract import enums as e, database as db
from pyalfred.contract.interface import DatabaseInterface


class FunctionDecorator(object):
    def __init__(self, f, task, include_task=True):
        self._task = task  # type: BaseTask
        self._f = f
        self._include_task = include_task

    def __call__(self, *args, **kwargs):
        try:
            self._task.status = e.Status.Running

            if self._include_task:
                kwargs["task_obj"] = self._task.db

            return self._f(*args, **kwargs)

        except Exception as exc:
            self._task.fail(exc)

            raise exc


class BaseTask(object):
    def __init__(self, f: Callable[[Tuple[Any], Dict[str, Any]], Any], intf: DatabaseInterface, args=None, kwargs=None):
        """
        Defines a base class for tasks.
        """

        self._f = FunctionDecorator(f, self)
        self._args = args
        self._kwargs = kwargs

        self._metas = dict()

        self._db = None  # type: db.Task
        self._intf = intf

    def initialize(self, key: str = None):
        task = db.Task(key=key or uuid4().hex, start_time=datetime.now(), end_time=datetime.max, status=e.Status.Queued)

        self._db = self._intf.create(task)
        return self

    @property
    def db(self) -> db.Task:
        return self._db

    @property
    def key(self) -> str:
        return self._db.key

    @property
    def status(self) -> e.Status:
        return self._db.status

    @status.setter
    def status(self, x: e.Status):
        self._db.status = x

        if x in (e.Status.Failed, e.Status.Done):
            self._db.end_time = datetime.now()

        self._db = self._intf.update(self._db)[0]

    def add_meta(self, key, value):
        if key not in self._metas:
            self._metas[key] = db.TaskMeta(task_id=self._db.id)

        self._metas[key].key = key
        self._metas[key].value = value

        return self

    def fail(self, exc: Exception):
        self._intf.create(db.TaskException(task_id=self._db.id, type_=exc.__class__.__name__, message=repr(exc)))

        self.status = e.Status.Failed

        return self

    def update_meta(self):
        # TODO: Improve
        for k, v in self._metas.items():
            if v.id is None:
                self._metas[k] = self._intf.create(v)
            else:
                self._metas[k] = self._intf.update(v)

        return self
