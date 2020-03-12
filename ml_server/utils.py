from flask_restful.reqparse import RequestParser
from pandas.util import hash_pandas_object
from hashlib import sha256
import pandas as pd
from .app import app
from werkzeug.exceptions import BadRequest
from .modelmanager import BaseModelManager


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

            app.logger.exception('Failed in task', e)
            return {'message': str(e)}, 500

    return wrap


def custom_login(auth):
    def wrap(func):
        def wrapp(*args, **kwargs):
            if app.config['EXTERNAL_AUTH']:
                return func(*args, **kwargs)

            return auth(func)(*args, **kwargs)

        return wrapp

    return wrap


def run_model(func, x, model_manager, key, **kwargs):
    """
    Utility function
    :param func: The function to apply
    :param x: The data
    :param model_manager: A model manager
    :type model_manager: BaseModelManager
    :param key: The key
    :param meta: The meta
    :return:
    """

    model_manager.pre_model_start(key)

    try:
        return func(x, **kwargs)
    except Exception as e:
        app.logger.exception(f'Failed task with key: {key}', e)
        model_manager.model_fail(key)
        raise e

