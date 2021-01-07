from .base import Saveable
from ..types import FrameOrArray
from .utils import container_exists


class UpdateSession(Saveable):
    def __init__(self, context, blueprint):
        super().__init__(context, blueprint)
        self._container = self._load()

    def _load(self):
        result = self.context.get_result()
        return self._blueprint.deserialize(result)

    @container_exists()
    def update(self, x: FrameOrArray, y: FrameOrArray = None, **kwargs):
        self._blueprint.fit(self._container, x, y, **kwargs)
