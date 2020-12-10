from rq import Queue
from datetime import datetime
from pylurch.contract import enums as e, database as db
from .base import BaseTask


class RQTask(BaseTask):
    """
    Class for tasking queues with 'RQ'.
    """

    def __init__(self, f, intf, args=None, kwargs=None):
        super().__init__(f, intf, args=args, kwargs=kwargs)
        self._timeout = 2 * 3600

    def make_rqtask(self, queue: Queue):
        return queue.create_job(self._f, args=self._args, kwargs=self._kwargs, timeout=self._timeout)

    def initialize(self, key: str = None):
        task = db.Task(key=key, start_time=datetime.now(), end_time=datetime.max, status=e.Status.Queued)

        self._db = self._intf.create(task)
        return self
