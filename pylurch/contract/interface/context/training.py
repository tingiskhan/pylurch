from typing import List, Dict, Any
from ...database import Label, Score, Package
from .update import UpdateContext


class TrainingContext(UpdateContext):
    def __init__(self, client, training_session):
        super().__init__(client, training_session, None)

    def add_label(self, label: str):
        self._to_commit.put(Label(session_id=self._session.id, label=label))

    def add_labels(self, labels: List[str]):
        for label in labels:
            self.add_label(label)

    def add_score(self, key: str, value: float):
        self._to_commit.put(Score(session_id=self._session.id, key=key, value=value))

    def add_scores(self, scores: Dict[str, Any]):
        for k, v in scores.items():
            self.add_score(k, v)

    def add_package(self, package: str, version: str):
        self._to_commit.put(Package(session_id=self._session.id, name=package, version=version))

    def add_packages(self, packages: Dict[str, str]):
        for k, v in packages.items():
            self.add_package(k, v)
