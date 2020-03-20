from flask_restful.reqparse import RequestParser
from pandas.util import hash_pandas_object
from hashlib import sha256
import pandas as pd
from werkzeug.exceptions import BadRequest
from .model_managers import BaseModelManager
from .app import ac


BASE_REQ = RequestParser()


def hash_series(*args):
    """
    Hash a number of pandas DataFrames.
    :param args: A number of Series/DataFrames
    :type args: tuple[Series]|tuple[DataFrame]
    :rtype: str
    """

    conc = pd.concat(args, axis=1)

    return sha256(hash_pandas_object(conc, index=True).values).hexdigest()


def custom_error(func):
    def wrap(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, BadRequest):
                raise e

            ac.app.logger.exception('Failed in task', e)
            return {'message': str(e)}, 500

    return wrap


def custom_login(auth):
    def wrap(func):
        def wrapp(*args, **kwargs):
            if ac.app.config['EXTERNAL_AUTH']:
                return func(*args, **kwargs)

            return auth(func)(*args, **kwargs)

        return wrapp

    return wrap


def run_model(func, model, x, model_manager, name, key, backend, **kwargs):
    """
    Utility function
    :param func: The function to apply
    :param model: The model
    :param x: The data
    :param model_manager: A model manager
    :type model_manager: BaseModelManager
    :param name: The name of the model
    :param key: The key
    :param backend: The backend
    :return:
    """

    model_manager.pre_model_start(name, key, backend)

    try:
        return func(model, x, **kwargs)
    except Exception as e:
        ac.app.logger.exception(f'Failed task with key: {key}', e)
        model_manager.model_fail(name, key, backend)
        raise e
