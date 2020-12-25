from typing import TypeVar, Generic

TModel = TypeVar("TModel")


class InferenceContainer(Generic[TModel]):
    def __init__(self, model: TModel):
        self._model = model

    @property
    def model(self) -> TModel:
        return self._model
