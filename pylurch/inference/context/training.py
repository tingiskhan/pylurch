from .base import SessionContext
from .utils import container_exists
from ..types import FrameOrArray


class TrainingSessionContext(SessionContext):
    def __init__(self, context, blueprint, **kwargs):
        super().__init__(context, blueprint)
        self._container = self.initialize(**kwargs)

    def initialize(self, **kwargs):
        return self._blueprint.make_model(**kwargs)

    def _on_exit(self):
        self._container = None

    @property
    @container_exists()
    def model(self):
        return self._container.model

    @property
    def container(self):
        return self._container

    @container_exists()
    def fit(self, x: FrameOrArray, y: FrameOrArray = None, **kwargs):
        self._blueprint.fit(self._container, x, y, **kwargs)

    @container_exists()
    def save(self, *args, x: FrameOrArray = None, y: FrameOrArray = None):
        artifacts = self._blueprint.serialize(self._container, *args, x=x, y=y)
        self.context.add_result(artifacts)
