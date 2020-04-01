import pandas as pd
from .utils import hash_series, custom_error
from hashlib import sha256
from .db.enums import ModelStatus, SerializerBackend, EXECUTOR_MAP
import numpy as np
from .resource import BaseModelResource


class ModelResource(BaseModelResource):
    def make_model(self, **kwargs):
        """
        Creates the model to be used.
        :param kwargs: Any key-worded arguments passed to the model.
        :rtype: BaseModel
        """

        raise NotImplementedError()

    def serializer_backend(self):
        """
        Returns the backend used by the backend as a string. See db.enums.SerializerBackends
        :rtype: ModelStatus
        """

        raise NotImplementedError()

    def done_callback(self, fut, key, x, **kwargs):
        """
        What to do when the callback is done.
        :param fut: The future
        :type fut: flask_executor.futures.Future
        :param key: The key associated with the model
        :type key: str
        :param x: The data
        :type x: pandas.DataFrame
        """

        try:
            res = fut.result()
            self.logger.info(f'Successfully trained {key}, now trying to persist')

            bytestring = self.serialize(res, x)

            meta_data = self.add_metadata(res, **kwargs)
            self.save_model(key, bytestring)

            self.logger.info(f'Successfully persisted {key}')
        except Exception as e:
            self.logger.exception(f'Failed persisting {key}', e)
            self.model_manager.model_fail(self.name(), key, self.serializer_backend())
        finally:
            self.executor.futures.pop(self._make_executor_key(key))

    def serialize(self, model, x, y=None):
        """
        Serialize model to byte string.
        :param model: The model to convert
        :param x: The data used for training
        :param y: The data used for training
        :return: onnx
        """

        raise NotImplementedError()

    def save_model(self, key, mod, meta_data=None):
        """
        Method for saving the model.
        :param key: The key
        :type key: str
        :param mod: The model
        :type mod: bytes
        :param meta_data: Whether to add any metadata associated with the model
        :type meta_data: pd.Series
        :return: Nothing
        :rtype: None
        """

        self.model_manager.save(self.name(), key, mod, self.serializer_backend())

    def load_model(self, key):
        """
        Method for loading the model.
        :param key: The key
        :type key: str
        :return: TrainingSession
        """

        obj = self.model_manager.load(self.name(), key, self.serializer_backend())

        return self._load(obj)

    def _load(self, obj):
        """
        To be overridden if you save something other than the actual model. E.g. if you choose to save a state
        dictionary you need be override this method.
        :param obj: The object
        :type obj: object
        :return: Model
        """

        return obj

    def fit(self, model, x, y=None, **kwargs):
        """
        Fits the model
        :param model: The model to use
        :param x: The data
        :type x: pd.DataFrame
        :param y: The response data (if any)
        :type y: pd.DataFrame
        :param kwargs: Any additional key worded arguments
        :return: The model
        """

        raise NotImplementedError()

    def update(self, model, x, y=None, **kwargs):
        """
        Fits the model
        :param model: The model to use
        :param x: The data
        :type x: pd.DataFrame
        :param y: The response data (if any)
        :type y: pd.DataFrame
        :param kwargs: Any additional key worded arguments
        :return: The model
        """

        raise ValueError('This model does not support updating!')

    def predict(self, mod, x, orient, **kwargs):
        """
        Return the prediction.
        :param mod: The model
        :param x: The data to predict for
        :type x: pd.DataFrame
        :param orient: The orientation
        :type orient: str
        :return: JSON like dict
        :rtype: dict
        """

        if self.serializer_backend() != SerializerBackend.ONNX:
            raise NotImplementedError(f'You must override the method yourself!')

        inp_name = mod.get_inputs()[0].name
        label_name = mod.get_outputs()[0].name

        res = mod.run([label_name], {inp_name: x.values.astype(np.float32)})[0]

        return {'y': pd.DataFrame(res, index=x.index, columns=['y']).to_json(orient=orient)}

    def run_model(self, func, model, x, key, **kwargs):
        """
        Utility function
        :param func: The function to apply
        :param model: The model
        :param x: The data
        :param name: The name of the model
        :param key: The key
        :return:
        """

        self.model_manager.pre_model_start(self.name(), key, self.serializer_backend())

        try:
            return func(model, x, **kwargs)
        except Exception as e:
            self.logger.exception(f'Failed task with key: {key}', e)
            self.model_manager.model_fail(self.name(), key, self.serializer_backend())
            raise e

    def parse_data(self, data, **kwargs):
        """
        Method for parsing data.
        :param data: The data in string format
        :type data: str
        :return: Data in required format
        """

        return pd.read_json(data, **kwargs).sort_index()

    def _make_executor_key(self, key):
        return f'{self.name()}-{key}'

    def check_model_status(self, key):
        """
        Helper function for checking the status of the model.
        :param key: The key of the model
        :type key: str
        :return: String indicating status
        :rtype: ModelStatus
        """

        running_in_executor = self.executor.futures._state(self._make_executor_key(key))

        if running_in_executor:
            return EXECUTOR_MAP[running_in_executor]

        return self.model_manager.check_status(self.name(), key, self.serializer_backend())

    def name(self):
        """
        The name of the model to use.
        :return: Name of the model
        :rtype: str
        """

        raise NotImplementedError()

    def add_metadata(self, model, **kwargs):
        """
        Allows user to add string meta data associated with the model. Is called when model is done.
        :param model: The instantiated model.
        :param kwargs: The key worded arguments associated with the model
        :return: A pandas.Series indexed with key -> value
        :rtype: pandas.Series
        """

        return self

    @custom_error
    def _put(self, **args):
        # ===== Get data ===== #
        orient = args['orient']
        x = self.parse_data(args['x'], orient=orient)

        retrain = args['retrain'].lower() == 'true'

        # ===== Define model ===== #
        modkwargs = args['modkwargs'] or dict()
        akws = args['algkwargs'] or dict()

        if args['y'] is not None:
            akws['y'] = self.parse_data(args['y'], orient=orient)

        model = self.make_model(**modkwargs)

        # ===== Generate model key ===== #
        dkey = hash_series(x) if args['name'] is None else sha256((args['name']).encode()).hexdigest()

        # ===== Check status ===== #
        status = self.check_model_status(dkey)

        if status == ModelStatus.Running:
            if retrain:
                return {
                    'message': f'Cannot cancel already {status} task. Try re-running when model is done',
                    'model-key': dkey
                }

            return {'message': status.value, 'model-key': dkey}

        if (status is not None and status != ModelStatus.Failed) and not retrain:
            self.logger.info('Model already exists, and no retrain requested')
            return {'message': status.value, 'model-key': dkey}

        key = self._make_executor_key(dkey)

        futures = self.executor.submit_stored(key, self.run_model, self.fit, model, x, dkey, **akws)
        futures.add_done_callback(lambda u: self.done_callback(u, dkey, x=x, **akws))

        self.logger.info(f'Successfully started training of {self.name()} using {x.shape[0]} observations')

        return {'model-key': dkey}

    @custom_error
    def _post(self, **args):
        key = args['model-key']

        status = self.check_model_status(key)

        if status is None:
            return {'message': 'TrainingSession does not exist!'}, 400

        if status != ModelStatus.Done:
            return {'message': status.value}

        mod = self.load_model(key)

        self.logger.info(f'Predicting values using model {self.name()}')

        return self.predict(mod, self.parse_data(args['x'], orient=args['orient']), orient=args['orient'])

    @custom_error
    def _get(self, key):
        status = self.check_model_status(key)

        if status != ModelStatus.Done:
            return {'message': status.value}

        mod = self.load_model(key)
        return self.get_return({'message': ModelStatus.Done}, mod)

    @custom_error
    def _patch(self, **args):
        dkey = args['model-key']

        status = self.check_model_status(dkey)

        if status is None:
            return {'message': 'TrainingSession does not exist!'}, 400

        if status != ModelStatus.Done:
            return {'message': status.value}

        model = self.load_model(dkey)
        x = self.parse_data(args['x'])

        kwargs = dict()
        if 'y' in args:
            kwargs['y'] = self.parse_data(args['y'])

        key = self._make_executor_key(dkey)

        futures = self.executor.submit_stored(key, self.run_model, self.update, model, x, dkey, **kwargs)
        futures.add_done_callback(lambda u: self.done_callback(u, dkey, x=x))

        self.logger.info(f'Started updating of model {self.name()} using {x.shape[0]} new observations')

        return {'message': self.executor.futures._state(dkey)}

    @custom_error
    def _delete(self, key):
        self.logger.info(f'Deleting model with key: {key}')

        self.model_manager.delete(self.name(), key, self.serializer_backend())

        self.logger.info(f'Successfully deleted model with key: {key}')

        return {'message': 'SUCCESS'}