from pandas.util import hash_pandas_object
from hashlib import sha256
import pandas as pd
from falcon.errors import HTTPBadRequest
from typing import Iterable, Callable, Dict
from pandas._typing import FrameOrSeries
from falcon.status_codes import HTTP_500
import logging


def hash_series(*args: Iterable[FrameOrSeries]) -> str:
    """
    Hash a number of pandas DataFrames.
    :param args: A number of Series/DataFrames
    """

    conc = pd.concat(args, axis=1)

    return sha256(hash_pandas_object(conc, index=True).values).hexdigest()


def custom_error(func: Callable[[object, Iterable, Dict], object]):
    def wrap(obj, *args, **kwargs):
        try:
            return func(obj, *args, **kwargs)
        except Exception as e:
            if isinstance(e, HTTPBadRequest):
                raise e

            obj.logger.exception('Failed in task', e)
            return {'message': str(e)}, HTTP_500

    return wrap


def make_base_logger(name: str) -> logging.Logger:
    """
    Helper method for generating a simple base logger.
    :param name: The name of the logging module
    """

    logger = logging.getLogger(name)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(name)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.setLevel(logging.DEBUG)

    return logger
