from pandas.util import hash_pandas_object
from hashlib import sha256
import pandas as pd
from werkzeug.exceptions import BadRequest
from typing import Iterable, Callable, Dict
from pandas._typing import FrameOrSeries


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
            if isinstance(e, BadRequest):
                raise e

            obj.logger.exception('Failed in task', e)
            return {'message': str(e)}, 500

    return wrap
