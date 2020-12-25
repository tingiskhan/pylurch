from redis import Redis
from rq import Queue
from pyalfred.contract.interface import DatabaseInterface
from pylurch.contract.enums import Status
from pylurch.contract.database import Task
from ..tasks import RQTask
from .base import BaseRunner


def _when_done(task_id: int, interface: DatabaseInterface):
    task = interface.get(Task, lambda u: u.id == task_id, one=True)

    if task.status != Status.Running:
        task.status = Status.Unknown
    else:
        task.status = Status.Done

    interface.update(task)


class RQRunner(BaseRunner):
    def __init__(self, conn: Redis, client, **kwargs):
        """
        Class for enqueuing tasks using 'RQ'.
        """

        super().__init__(client)
        self._conn = conn
        self._queue = Queue(connection=conn, **kwargs)

    def _enqueue(self, task: RQTask):
        rq_task = task.make_rqtask(self._queue)
        task.initialize(rq_task.id)

        self._queue.enqueue_job(rq_task)
        self._queue.enqueue(_when_done, task.db.id, self._client, depends_on=rq_task)

    def make_task(self, f, *args, **kwargs) -> RQTask:
        return RQTask(f, self._client, args=args, kwargs=kwargs)

    def get_result(self, task_id):
        job = self._queue.fetch_job(task_id)

        if job is None:
            return None

        return job.result
