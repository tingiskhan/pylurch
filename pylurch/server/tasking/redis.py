from .task import RQTask
from .wrapper import BaseWrapper
from redis import Redis
from rq import Queue
from pylurch.contract.enums import Status


class RQWrapper(BaseWrapper):
    def __init__(self, conn: Redis, interface, **kwargs):
        """
        Class for enqueuing tasks using 'RQ'.
        """

        super().__init__(interface)
        self._conn = conn
        self._queue = Queue(connection=conn, **kwargs)

    def _enqueue(self, task: RQTask):
        rqtask = task.make_rqtask(self._queue)
        task.initialize(rqtask.id)

        self._queue.enqueue_job(rqtask)

    def make_task(self, f, *args, **kwargs) -> RQTask:
        return RQTask(f, self._i, args=args, kwargs=kwargs)

    def check_status(self, task_id):
        job = self._queue.fetch_job(task_id)

        if job is None:
            db_res = super(RQWrapper, self).check_status(task_id)

            if db_res is not None:
                return db_res

            return Status.Unknown

        status = job.get_status()
        if status == "queued":
            return Status.Queued

        if status == "started":
            return Status.Running

        if status == "finished":
            return Status.Done

        if status == "failed":
            return Status.Failed

        return Status.Unknown

    def get_result(self, task_id):
        job = self._queue.fetch_job(task_id)

        if job is None:
            return None

        return job.result
