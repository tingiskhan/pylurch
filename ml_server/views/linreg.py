from .model import ModelResource
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LinearRegression


class LinearRegressionView(ModelResource):
    def convert_to_onnx(self, model, x, y=None):

        inputs = [
            ('x', FloatTensorType([None, x.shape[-1]])),
        ]

        return convert_sklearn(model, 'linear_regression', inputs)

    def fit(self, model, x, y=None, **kwargs):
        return model.fit(x, y)

    def make_model(self, **kwargs):
        return LinearRegression(**kwargs)