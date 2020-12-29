import pandas as pd
from typing import Dict, Tuple, Any, Generic, TypeVar
import numpy as np
import git
from pylurch.contract import database as db, enums
from .container import InferenceContainer, LoadedContainer, TModel
from .types import FrameOrArray


TOutput = TypeVar("TOutput")
T = InferenceContainer[TModel]
U = LoadedContainer[TOutput]


class InferenceModelBlueprint(Generic[TModel, TOutput]):
    def get_revision(self):
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha

    def name(self):
        raise NotImplementedError()

    def make_model(self, **kwargs) -> T:
        raise NotImplementedError()

    def serialize(
        self, container: T, *args: Tuple[Any, ...], x: FrameOrArray = None, y: FrameOrArray = None
    ) -> Tuple[db.Artifact, ...]:
        raise NotImplementedError()

    def deserialize(self, artifacts: Tuple[db.Artifact, ...]) -> U:
        raise NotImplementedError()

    def fit(self, container: T, x: FrameOrArray, y: FrameOrArray = None, **kwargs: Dict[str, object]):
        raise NotImplementedError()

    def update(self, container: T, x: FrameOrArray, y: FrameOrArray = None, **kwargs: Dict[str, object]):
        raise ValueError("This model does not support updating!")

    def predict(self, container: U, x: FrameOrArray, **kwargs: Dict[str, object]) -> FrameOrArray:
        if container.backend != enums.Backend.ONNX:
            raise NotImplementedError(
                f"Backend must be of type of {enums.Backend.ONNX}, not {container.backend}"
            )

        inp_name = container.model.get_inputs()[0].name
        label_name = container.model.get_outputs()[0].name

        res = container.model.run([label_name], {inp_name: x.values.astype(np.float32)})[0]

        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(res, index=x.index, columns=["y"])

        return res
