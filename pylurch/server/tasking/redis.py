from .task import RQTask
from .wrapper import BaseWrapper
from redis import Redis
from rq import Queue, job as j


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

    def get_result(self, task_id):
        job = j.Job.fetch(task_id)

        if job.get_status() == "finished":
            return job.result

        return None
