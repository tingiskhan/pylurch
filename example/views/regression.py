from ml_api import ModelResource
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from ml_api.db.enums import SerializerBackend


class LinearRegressionView(ModelResource):
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


class LogisticRegressionView(LinearRegressionView):
    def name(self):
        return 'logistic-regression'

    def make_model(self, **kwargs):
        return LogisticRegressionCV(**kwargs)
