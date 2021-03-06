from pylurch.inference import ONNXModelBluePrint, InferenceContainer
from pylurch.contract.database import Artifact
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from pylurch.contract.enums import Backend, ArtifactType


class LinearRegressionBlueprint(ONNXModelBluePrint[LinearRegression]):
    def name(self):
        return "linear-regression-model"

    def serialize(self, container, *_, x=None, y=None):
        inputs = [
            ("x", FloatTensorType([None, x.shape[-1]])),
        ]

        return (
            Artifact(
                type_=ArtifactType.Model,
                backend=Backend.ONNX,
                bytes=convert_sklearn(container.model, self.name(), inputs).SerializeToString(),
            ),
        )

    def make_model(self, **kwargs):
        return InferenceContainer(LinearRegression(**kwargs))

    def fit(self, container: InferenceContainer[LinearRegression], x, y=None, **kwargs):
        container.model.fit(x, y)
        return container

    def deserialize(self, artifacts):
        if len(artifacts) > 1:
            raise ValueError("Cannot handle more than one artifact!")

        return InferenceContainer(rt.InferenceSession(artifacts[0].bytes))


class LogisticRegressionBlueprint(LinearRegressionBlueprint):
    def name(self):
        return "logistic-regression-model"

    def make_model(self, **kwargs):
        return InferenceContainer(LogisticRegressionCV(**kwargs))
