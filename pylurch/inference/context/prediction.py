from .base import SessionContext
from ..types import FrameOrArray


class PredictionSessionContext(SessionContext):
    def __init__(self, context, blueprint):
        super().__init__(context, blueprint)
        self._container = self._load()

    @property
    def container(self):
        return self._container

    def _load(self):
        result = self.context.get_result()
        return self._blueprint.deserialize(result)

    def _on_exit(self):
        self._container = None

    def predict(self, x: FrameOrArray, **kwargs):
        return self._blueprint.predict(self._container, x, **kwargs)
