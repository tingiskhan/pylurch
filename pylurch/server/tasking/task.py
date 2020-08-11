from typing import Dict, Callable, Tuple, Any
from pylurch.contract import enums as e, database as db, interfaces as i
from datetime import datetime
from uuid import uuid4
from rq import Queue


class Decorator(object):
    def __init__(self, f, task):
        self._task = task
        self._f = f

    def __call__(self, *args, **kwargs):
        try:
            # ===== Try getting result ===== #
            self._task.status = e.Status.Running
            res = self._f(*args, **kwargs)

            # ===== Update task status ===== #
            self._task.status = e.Status.Done
        except Exception as exc:
            self._task.status = e.Status.Failed


class TaskWrapper(object):
    def __init__(self, f: Callable[[Tuple[Any], Dict[str, Any]], Any], intf: i.DatabaseInterface, args=None,
                 kwargs=None):
        """
        Defines a base class for tasks.
        """

        self._f = Decorator(f, self)
        self._args = args
        self._kwargs = kwargs

        self._metas = dict()

        self._db = None  # type: db.Task
        self._intf = intf

    def initialize(self, key: str = None):
        task = db.Task(
            key=key or uuid4().hex,
            start_time=datetime.now(),
            end_time=datetime.max,
            status=e.Status.Queued
        )

        self._db = self._intf.make_interface(db.Task).create(task)
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

        self._db = self._intf.make_interface(db.Task).update(self._db)[0]

    def add_meta(self, key, value):
        if key not in self._metas:
            self._metas[key] = db.TaskMeta(task_id=self._db.id)

        self._metas[key].key = key
        self._metas[key].value = value

        return self

    def update_meta(self):
        # TODO: Improve
        intf = self._intf.make_interface(db.Task)
        for k, v in self._metas.items():
            if v.id is None:
                self._metas[k] = intf.create(v)
            else:
                self._metas[k] = intf.update(v)

        return self


class RQTask(TaskWrapper):
    """
    Class for tasking queues with 'RQ'.
    """

    def __init__(self, f, intf):
        super().__init__(f, intf)
        self._timeout = 2 * 3600

    def make_rqtask(self, queue: Queue):
        return queue.create_job(self._f, args=self._args, kwargs=self._kwargs, timeout=self._timeout)

    def initialize(self, key: str = None):
        task = db.Task(
            key=key,
            start_time=datetime.now(),
            end_time=datetime.max,
            status=e.Status.Queued
        )

        self._db = self._intf.make_interface(db.Task).create(task)
        return self

