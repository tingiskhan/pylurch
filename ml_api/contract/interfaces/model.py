import requests as r
from time import sleep
import pandas as pd
from .base import BaseInterface
from ..schemas import GetResponse, PutResponse, PostResponse
from ..enums import ModelStatus
import numpy as np
import json


class GenericModelInterface(BaseInterface):
    def __init__(self, base, endpoint, **modkwargs):
        """
        Implements an interface for talking to models.
        :param modkwargs: Any key worded arguments for the model to pass on instantiation
        """

        super().__init__(base, endpoint)
        self._mk = modkwargs

        self._key = None
        self._orient = 'columns'

    def load(self, key: str):
        """
        Loads the model with specified key.
        :param key: The key of the model to load
        """

        self._key = key

        return self

    @property
    def model_key(self) -> str:
        return self._key

    @property
    def _address(self) -> str:
        return f'{self._base}/{self._ep}'

    def _check_done(self):
        resp = GetResponse().load(self._exec_req(r.get, params={'model_key': self._key}))

        if resp['status'] == ModelStatus.Failed:
            raise Exception('Something went wrong trying to train the model! Check server logs for further details')

        return resp['status'] == ModelStatus.Done

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None, name: str = None, wait: bool = True, **algkwargs):
        """
        Method for fitting the model.
        :param x: The x-data
        :param y: The y-data (if any)
        :param name: Whether to name the data
        :param wait: Whether to wait for it complete
        :param algkwargs: Any algorithm key words
        """

        params = {
            'x': x.to_json(orient=self._orient),
            'algkwargs': algkwargs,
            'modkwargs': self._mk,
            'orient': self._orient,
            'retrain': algkwargs.pop('retrain', False)
        }

        if y is not None:
            params['y'] = y.to_json(orient=self._orient)
        if name:
            params['name'] = name

        return self._train(r.put, wait=wait, json=params)

    def _train(self, meth: callable, wait: bool = False, **kwargs):
        resp = meth(self._address, **kwargs)

        if resp.status_code != 200:
            raise Exception(f'Got code {resp.status_code}: {resp.text}')

        self._key = PutResponse().load(resp.json())['model_key']

        while wait and not self._check_done():
            sleep(5)

        return self

    def predict(self, x: pd.DataFrame, as_array=False, **kwargs):
        """
        Predict the model.
        :param x: The DataFrame to predict for
        :param as_array: Whether to return as an numpy.array or pandas.DataFrame. Returning large DataFrames take
        considerably longer time than a corresponding sized numpy.ndarray.
        """

        if not self._key:
            raise ValueError('Must call `fit` first!')

        params = {
            'model_key': self._key,
            'x': x.to_json(orient=self._orient),
            'orient': self._orient,
            'as_array': as_array,
            'kwargs': kwargs or dict()
        }

        resp = PostResponse().load(self._exec_req(r.post, json=params))
        data = json.loads(resp['data'])

        if not as_array:
            return pd.DataFrame.from_dict(data, orient=resp['orient'])

        return np.array(data)

    def update(self, x: pd.DataFrame, y: pd.DataFrame = None, wait: bool = True):
        """
        Updates the model. See docs of `fit` for docs pertaining to parameters.
        """

        params = {
            'x': x.to_json(orient=self._orient),
            'orient': self._orient,
            'model_key': self._key
        }

        if y is not None:
            params['y'] = y.to_json(orient=self._orient)

        return self._train(r.patch, wait=wait, json=params)

    def delete(self, model_key: str):
        """
        Deletes all instances of a model with the model key.
        :param model_key: The model key
        """

        req = r.delete(self._address, params={'model_key': model_key})

        return self