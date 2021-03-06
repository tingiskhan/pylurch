from typing import Dict, Tuple, Any, Generic, TypeVar
import git
import pandas as pd
import numpy as np
from pylurch.contract import database as db
from onnxruntime import InferenceSession
from .container import InferenceContainer, TModel
from .types import FrameOrArray


TOutput = TypeVar("TOutput")
T = InferenceContainer[TModel]
U = InferenceContainer[TOutput]


class InferenceModelBlueprint(Generic[TModel, TOutput]):
    def get_revision(self) -> str:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha

    def name(self) -> str:
        raise NotImplementedError()

    def make_model(self, **kwargs) -> T:
        raise NotImplementedError()

    def serialize(
        self, container: T, *args: Tuple[Any, ...], x: FrameOrArray = None, y: FrameOrArray = None
    ) -> Tuple[db.Artifact, ...]:
        raise NotImplementedError()

    def deserialize(self, artifacts: Tuple[db.Artifact, ...]) -> U:
        raise NotImplementedError()

    def fit(self, container: T, x: FrameOrArray, y: FrameOrArray = None, **kwargs: Dict[str, object]) -> T:
        raise NotImplementedError()

    def update(self, container: U, x: FrameOrArray, y: FrameOrArray = None, **kwargs: Dict[str, object]) -> U:
        raise ValueError()

    def predict(self, container: U, x: FrameOrArray, **kwargs: Dict[str, object]) -> FrameOrArray:
        raise NotImplementedError()


class ONNXModelBluePrint(InferenceModelBlueprint[TModel, InferenceSession]):
    def predict(self, container: T, x: FrameOrArray, **kwargs: Dict[str, object]):
        inp_name = container.model.get_inputs()[0].name
        label_name = container.model.get_outputs()[0].name

        res = container.model.run([label_name], {inp_name: x.values.astype(np.float32)})[0]

        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(res, index=x.index, columns=["y"])

        return res