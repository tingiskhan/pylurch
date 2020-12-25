from typing import List, Dict, Any
from queue import Queue
from ...database import Result, Label, Score, Package
from ...enums import SerializerBackend
from .base import ClientContext


class ClientTrainingContext(ClientContext):
    def __init__(self, client, training_session):
        super().__init__(client, training_session)

        self._to_commit = Queue()

    def add_label(self, label: str):
        return Label(session_id=self._session.id, label=label)

    def add_labels(self, labels: List[str]):
        for label in labels:
            self.add_label(label)

    def add_score(self, key: str, value: float):
        self._to_commit.put(Score(session_id=self._session.id, key=key, value=value))

    def add_scores(self, scores: Dict[str, Any]):
        for k, v in scores.items():
            self.add_score(k, v)

    def add_result(self, model: bytes, backend: SerializerBackend):
        self._to_commit.put(Result(session_id=self._session.id, bytes=model, backend=backend))

    def add_package(self, package: str, version: str):
        self._to_commit.put(Package(session_id=self._session.id, name=package, version=version))

    def add_packages(self, packages: Dict[str, str]):
        for k, v in packages.items():
            self.add_package(k, v)

    def on_exit(self):
        for to_commit in iter(self._to_commit.get, None):
            self._client.create(to_commit)

            if self._to_commit.empty():
                return
