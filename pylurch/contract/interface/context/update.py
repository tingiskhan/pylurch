from queue import Queue
from typing import Sequence
from .prediction import PredictionContext
from ...database import Artifact, TrainingSession


# TODO: Should inherit from both
class UpdateContext(PredictionContext):
    def __init__(self, client, training_session, old_session: TrainingSession):
        super().__init__(client, training_session)
        self._old_session = old_session
        self._to_commit = Queue()

    def get_result(self):
        return self._get_result(self._old_session.id)

    def add_artifact(self, artifact: Artifact):
        artifact.session_id = self.session.id
        self._to_commit.put(artifact)

    def add_artifacts(self, artifacts: Sequence[Artifact]):
        for a in artifacts:
            self.add_artifact(a)

    def on_exit(self):
        for to_commit in iter(self._to_commit.get, None):
            self._client.create(to_commit, batched=True)

            if self._to_commit.empty():
                return
