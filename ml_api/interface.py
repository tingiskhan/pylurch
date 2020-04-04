import requests as r


class ModelInterface(object):
    def __init__(self, address, endpoint, **modkwargs):
        """
        An interface for training and predicting models.
        :param address: The address to the server
        :type address: str
        :param endpoint: The endpoint of the model
        :type endpoint: str
        """

        self._addr = f'{address if not address.endswith("/") else address[:-1]}/{endpoint}'
        self._modkwargs = modkwargs
        self._mk = None
        self._orient = 'columns'

        self._headers = {
            'Content-type': 'application/json'
        }

    def add_headers(self, dct):
        """
        Adds headers.
        :param dct: Dictionary
        :type dct: dict[str, str]
        :return: Self
        :rtype: ModelInterface
        """

        self._headers.update(dct)

    def fit(self, x, y=None, retrain=False, **algkwargs):
        """
        Fits the model using the data.
        :param x:
        :param y:
        :param retrain:
        :return:
        """

        params = {
            'x': x.to_json(orient=self._orient),
            'algkwargs': algkwargs,
            'retrain': retrain
        }

        if y is not None:
            params['y'] = y.to_json(orient=self._orient)

        req = r.post(self._addr, headers=self._headers, json=params)

        if req.status_code != 200:
            raise ValueError(f'Something went wrong: {req.text}')

        self._mk = req.json()['model-key']

        return self

    def predict(self, x):
        """
        Predicts the model.
        :param x:
        :return:
        """

    def delete(self):
        """
        Delete instances of model.
        :return:
        """

    def update(self, x):
        """
        Updates the model with new data.
        :param x:
        :return:
        """