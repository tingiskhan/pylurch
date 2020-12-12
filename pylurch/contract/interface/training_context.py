from typing import List, Dict, Any
from pyalfred.contract.interface import DatabaseInterface
from ..database import TrainingSession, TrainingResult, Label, MetaData, Package
from ..enums import SerializerBackend


def decorator(key: str, exists):
    def wrapper(f):
        def dec(self, *args):
            if key not in self._to_commit:
                self._to_commit[key] = list()

            existing = next((item for item in self._to_commit[key] if exists(item, *args)), None)
            if existing is not None:
                self._to_commit[key].remove(existing)

            self._to_commit[key].append(f(self, *args))

            return

        return dec

    return wrapper


class TrainingContext(object):
    def __init__(self, intf: DatabaseInterface, training_session: TrainingSession):
        self._intf = intf
        self._session = training_session

        self._to_commit = dict()
        self._to_update = dict()

    @property
    def session(self):
        return self._session

    def add_labels(self, labels: List[str]):
        for label in labels:
            self.add_label(label)

    @decorator("labels", lambda item, label: item.label == label)
    def add_label(self, label: str):
        return Label(session_id=self._session.id, label=label)

    def add_metadatas(self, meta_data: Dict[str, Any]):
        for k, v in meta_data.items():
            self.add_metadata(k, v)

    @decorator("meta_data", lambda item, key, *args: item.key == key)
    def add_metadata(self, key: str, value: Any):
        return MetaData(session_id=self._session.id, key=key, value=str(value))

    @decorator("result", lambda *args: True)
    def add_result(self, model: bytes, backend: SerializerBackend):
        return TrainingResult(session_id=self._session.id, bytes=model, backend=backend)

    def add_packages(self, packages: Dict[str, str]):
        for k, v in packages.items():
            self.add_package(k, v)

    @decorator("packages", lambda item, package, *args: item.name == package)
    def add_package(self, package: str, version: str):
        return Package(session_id=self._session.id, name=package, version=version)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise Exception("Something went wrong during the session, see full error.") from exc_val

        for to_update in self._to_update.values():
            if any(to_update):
                self._intf.update(to_update)

        for to_commit in self._to_commit.values():
            if any(to_commit):
                self._intf.create(to_commit)

        return True
