from pylurch.inference import InferenceModelBlueprint, InferenceContainer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from pylurch.contract.enums import SerializerBackend


class LinearRegressionBlueprint(InferenceModelBlueprint[LinearRegression, rt.InferenceSession]):
    def serializer_backend(self):
        return SerializerBackend.ONNX

    def serialize(self, container, *_, x=None, y=None):
        inputs = [
            ("x", FloatTensorType([None, x.shape[-1]])),
        ]

        return convert_sklearn(container.model, self.name(), inputs).SerializeToString()

    def make_model(self, **kwargs):
        return InferenceContainer(LinearRegression(**kwargs))

    def fit(self, container: InferenceContainer[LinearRegression], x, y=None, **kwargs):
        container.model.fit(x, y)
        return container


class LogisticRegressionBlueprint(LinearRegressionBlueprint):
    def make_model(self, **kwargs):
        return InferenceContainer(LogisticRegressionCV(**kwargs))
