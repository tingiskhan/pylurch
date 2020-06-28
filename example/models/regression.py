from ml_api.inference import InferenceModel
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from ml_api.enums import SerializerBackend


class LinearRegressionModel(InferenceModel):
    def serializer_backend(self):
        return SerializerBackend.ONNX

    def serialize(self, model, x, y=None):
        inputs = [
            ('x', FloatTensorType([None, x.shape[-1]])),
        ]

        return convert_sklearn(model, self.name(), inputs).SerializeToString()

    def fit(self, model, x, y=None, **kwargs):
        return model.fit(x, y)

    def make_model(self, **kwargs):
        return LinearRegression(**kwargs)

    def name(self):
        return 'linear-regression'

    def add_metadata(self, model, **kwargs):
        return {
            'score': model.score(kwargs['x'], kwargs['y'])
        }


class LogisticRegressionModel(LinearRegressionModel):
    def name(self):
        return 'logistic-regression'

    def make_model(self, **kwargs):
        return LogisticRegressionCV(**kwargs)
