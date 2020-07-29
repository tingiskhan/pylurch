import requests as r
from time import sleep
import pandas as pd
from .base import BaseInterface
from ..schemas import GetResponse, PutResponse, PostResponse
from ..enums import Status
import numpy as np
import json


class GenericModelInterface(BaseInterface):
    def __init__(self, base, endpoint, **modkwargs):
        """
        Implements an interface for talking to models.
        :param modkwargs: Any key worded arguments for the model to pass on instantiation. Only applies to training
        """

        super().__init__(base, endpoint)
        self._mk = modkwargs

        self._name = None
        self._task_id = None
        self._orient = 'columns'

    def load(self, name: str):
        """
        Loads the model with specified key.
        :param name: The name of the model to load
        """

        self._name = name

        return self

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_done(self) -> bool:
        return self._check_done()

    @property
    def _address(self) -> str:
        return f'{self._base}/{self._ep}'

    def _check_done(self):
        if self._task_id is None:
            raise ValueError(f"Cannot check the status when 'task_id' is set to 'None'!")

        resp = GetResponse().load(self._exec_req(r.get, params={'task_id': self._task_id}))

        if resp['status'] == Status.Failed:
            raise Exception('Something went wrong trying to train the model! Check server logs for further details')

        return resp['status'] == Status.Done

    def fit(self, x: pd.DataFrame, session_name: str, y: pd.DataFrame = None, wait: bool = True, **algkwargs):
        """
        Method for fitting the model.
        :param x: The x-data
        :param y: The y-data (if any)
        :param session_name: The name of the session
        :param wait: Whether to wait for it complete
        :param algkwargs: Any algorithm key words
        """

        params = {
            'x': x.to_json(orient=self._orient),
            'algkwargs': algkwargs,
            'modkwargs': self._mk,
            'orient': self._orient,
            'name': session_name
        }

        if y is not None:
            params['y'] = y.to_json(orient=self._orient)

        return self._train(r.put, wait=wait, json=params)

    def _train(self, meth: callable, wait: bool = False, **kwargs):
        resp = meth(self._address, **kwargs)

        if resp.status_code != 200:
            raise Exception(f'Got code {resp.status_code}: {resp.text}')

        resp = PutResponse().load(resp.json())
        self._name = resp['session_name']

        if resp['status'] == Status.Done:
            return self

        self._task_id = resp['task_id']

        while wait and not self.is_done:
            sleep(5)

        return self

    def predict(self, x: pd.DataFrame, as_array=False, **kwargs):
        """
        Predict the model.
        :param x: The DataFrame to predict for
        :param as_array: Whether to return as an numpy.array or pandas.DataFrame. Returning large DataFrames take
        considerably longer time than a corresponding sized numpy.ndarray.
        """

        if not self._name:
            raise ValueError('Must call `fit` or `load` first!')

        params = {
            'name': self._name,
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

    def update(self, x: pd.DataFrame, session_name: str, y: pd.DataFrame = None, wait: bool = True):
        """
        Updates the model. See docs of `fit` for docs pertaining to parameters.
        """

        params = {
            'x': x.to_json(orient=self._orient),
            'orient': self._orient,
            'old_name': self._name,
            'name': session_name
        }

        if y is not None:
            params['y'] = y.to_json(orient=self._orient)

        return self._train(r.patch, wait=wait, json=params)