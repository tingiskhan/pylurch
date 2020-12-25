from typing import Union
from ..blueprint import InferenceModelBlueprint
from pylurch.contract.interface.context import ClientPredictionContext, ClientTrainingContext

T = Union[ClientPredictionContext, ClientTrainingContext]


# TODO: Move context?
class SessionContext(object):
    def __init__(self, context: T, blueprint: InferenceModelBlueprint):
        self._context = context
        self._blueprint = blueprint

    @property
    def context(self):
        return self._context

    @property
    def blueprint(self):
        return self._blueprint

    def _on_exit(self):
        return

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context.__exit__(exc_type, exc_val, exc_tb)
        self._on_exit()
        return True
