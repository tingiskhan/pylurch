from typing import TypeVar, Generic
from pylurch.contract.enums import Backend

TModel = TypeVar("TModel")


class InferenceContainer(Generic[TModel]):
    def __init__(self, model: TModel):
        self._model = model

    @property
    def model(self) -> TModel:
        return self._model


class LoadedContainer(InferenceContainer[TModel]):
    def __init__(self, model: TModel, backend: Backend):
        super().__init__(model)
        self.backend = backend
