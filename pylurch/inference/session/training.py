from .base import Saveable
from .utils import container_exists
from ..types import FrameOrArray


class TrainingSession(Saveable):
    def __init__(self, context, blueprint, **kwargs):
        super().__init__(context, blueprint)
        self._container = self.initialize(**kwargs)

    def initialize(self, **kwargs):
        return self._blueprint.make_model(**kwargs)

    def _on_exit(self):
        self._container = None

    @container_exists()
    def fit(self, x: FrameOrArray, y: FrameOrArray = None, **kwargs):
        self._blueprint.fit(self._container, x, y, **kwargs)
