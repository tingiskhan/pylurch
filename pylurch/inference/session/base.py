from typing import Union
from ..blueprint import InferenceModelBlueprint
from pylurch.contract.client.context import PredictionContext, TrainingContext, UpdateContext
from .utils import container_exists
from ..types import FrameOrArray


T = Union[PredictionContext, TrainingContext, UpdateContext]


# TODO: Move context?
class Session(object):
    def __init__(self, context: T, blueprint: InferenceModelBlueprint):
        self._context = context
        self._blueprint = blueprint
        self._container = None

    @property
    def context(self):
        return self._context

    @property
    def blueprint(self):
        return self._blueprint

    @property
    def container(self):
        return self._container

    def _on_exit(self):
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context.__exit__(exc_type, exc_val, exc_tb)
        self._on_exit()
        return True


class Saveable(Session):
    @container_exists()
    def save(self, *args, x: FrameOrArray = None, y: FrameOrArray = None):
        artifacts = self._blueprint.serialize(self._container, *args, x=x, y=y)
        self.context.add_artifacts(artifacts)

    @property
    @container_exists()
    def model(self):
        return self._container.model
