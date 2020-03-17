from flask_restful import Resource
import pandas as pd
from ..utils import BASE_REQ, hash_series, custom_error, custom_login, run_model
from ..app import executor, auth_token, app, MODEL_MANAGER
from hashlib import sha256
from ..db.enums import ModelStatus, SerializerBackend
import numpy as np


base_parser = BASE_REQ.copy()
base_parser.add_argument('x', type=str, required=True, help='JSON of data')
base_parser.add_argument('orient', type=str, help='The orientation of the JSON of the data', required=True)

patch_parser = base_parser.copy()
patch_parser.add_argument('y', type=str, help='The response variable')

put_parser = patch_parser.copy()
put_parser.add_argument('name', type=str, help='Name of the data set, used as key if provided')
put_parser.add_argument('modkwargs', type=dict, help='Kwargs for model instantiation')
put_parser.add_argument('algkwargs', type=dict, help='Kwargs for algorithm')
put_parser.add_argument('retrain', type=str, help='Whether to retrain using other data', default='False')

get_parser = BASE_REQ.copy()
get_parser.add_argument('model-key', type=str, required=True, help='Key of the model')

patch_parser.add_argument('model-key', type=str, required=True, help='Key of the model')

post_parser = patch_parser.copy()
post_parser.add_argument('model-key', type=str, required=True, help='Key of the model')


class ModelResource(Resource):
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
        :param x: The data
        :type x: pandas.DataFrame
        :type key: str
        """
        res = fut.result()
        app.logger.info(f'Successfully trained {key}, now trying to persist')

        try:
            bytestring = self.serialize(res, x)

            meta_data = self.add_metadata(res, **kwargs)
            self.save_model(MODEL_MANAGER, key, bytestring)

            app.logger.info(f'Successfully persisted {key}')
        except Exception as e:
            app.logger.exception(f'Failed persisting {key}', e)
            MODEL_MANAGER.model_fail(self.name(), key, self.serializer_backend())
        finally:
            executor.futures.pop(key)

    def serialize(self, model, x, y=None):
        """
        Serialize model to byte string.
        :param model: The model to convert
        :param x: The data used for training
        :param y: The data used for training
        :return: onnx
        """

        raise NotImplementedError()

    def save_model(self, model_manager, key, mod, meta_data=None):
        """
        Method for saving the model.
        :param model_manager: The model manager
        :type model_manager: ModelManager
        :param key: The key
        :type key: str
        :param mod: The model
        :type mod: bytes
        :param meta_data: Whether to add any metadata associated with the model
        :type meta_data: pd.Series
        :return: Nothing
        :rtype: None
        """

        model_manager.save(self.name(), key, mod, self.serializer_backend())

    def load_model(self, model_manager, key):
        """
        Method for loading the model.
        :param model_manager: The model manager
        :type model_manager: ModelManager
        :param key: The key
        :type key: str
        :return: TrainingSession
        """

        obj = model_manager.load(self.name(), key, self.serializer_backend())

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
        :rtype: str
        """

        running_in_executor = executor.futures._state(self._make_executor_key(key))

        if running_in_executor:
            return running_in_executor

        return MODEL_MANAGER.check_status(self.name(), key, self.serializer_backend())

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

    @custom_login(auth_token.login_required)
    @custom_error
    def put(self):
        args = put_parser.parse_args()

        # ===== Get data ===== #
        orient = args['orient']
        x = self.parse_data(args['x'], orient=orient)

        retrain = args['retrain'].lower() == 'true'

        # ===== Define model ===== #
        modkwargs = args['modkwargs'] or dict()
        akws = args['algkwargs'] or dict()

        if 'y' in args:
            akws['y'] = self.parse_data(args['y'])

        model = self.make_model(**modkwargs)

        # ===== Generate model key ===== #
        data_key = hash_series(x) if args['name'] is None else sha256((args['name']).encode()).hexdigest()

        # ===== Check status ===== #
        status = self.check_model_status(data_key)

        if status == ModelStatus.Running.value:
            if retrain:
                return {
                    'message': f'Cannot cancel already {status} task. Try re-running when model is done',
                    'model-key': data_key
                }

            return {'message': status, 'model-key': data_key}

        if status is not None and not retrain:
            app.logger.info('Model already exists, and no retrain requested')
            return {'message': status, 'model-key': data_key}

        key = self._make_executor_key(data_key)

        futures = executor.submit_stored(
            key, run_model, self.fit, model, x, MODEL_MANAGER, self.name(), data_key, self.serializer_backend(), **akws
        )

        futures.add_done_callback(lambda u: self.done_callback(u, data_key, x, **akws))

        app.logger.info(f'Successfully started training of {self.name()} using {x.shape[0]} observations')

        return {'model-key': data_key}

    @custom_login(auth_token.login_required)
    @custom_error
    def post(self):
        args = post_parser.parse_args()
        key = args['model-key']

        status = self.check_model_status(key)

        if status is None:
            return {'message': 'TrainingSession does not exist!'}, 400

        if status != ModelStatus.Done.value:
            return {'message': status}

        mod = self.load_model(MODEL_MANAGER, key)

        app.logger.info(f'Predicting values using model {self.name()}')

        return self.predict(mod, pd.read_json(args['x'], orient=args['orient']), orient=args['orient'])

    @custom_login(auth_token.login_required)
    @custom_error
    def get(self):
        args = get_parser.parse_args()
        key = args['model-key']

        status = self.check_model_status(key)

        if status != ModelStatus.Done.value:
            return {'message': status}

        mod = self.load_model(MODEL_MANAGER, key)
        return self.get_return({'message': 'DONE'}, mod)

    @custom_login(auth_token.login_required)
    @custom_error
    def patch(self):
        args = patch_parser.parse_args()
        dkey = args['model-key']

        status = self.check_model_status(dkey)

        if status is None:
            return {'message': 'TrainingSession does not exist!'}, 400

        if status != ModelStatus.Done.value:
            return {'message': status}

        model = self.load_model(MODEL_MANAGER, dkey)
        x = self.parse_data(args['x'])

        kwargs = dict()
        if 'y' in args:
            kwargs['y'] = self.parse_data(args['y'])

        key = self._make_executor_key(dkey)

        futures = executor.submit_stored(
            key, run_model, self.update, model, x, MODEL_MANAGER, self.name(), dkey, self.serializer_backend(), **kwargs
        )

        futures.add_done_callback(lambda u: self.done_callback(u, dkey, x))

        app.logger.info(f'Started updating of model {self.name()} using {x.shape[0]} new observations')

        return {'message': executor.futures._state(dkey)}

    @custom_login(auth_token.login_required)
    @custom_error
    def delete(self):
        args = get_parser.parse_args()
        key = args['model-key']

        app.logger.info(f'Deleting model with key: {key}')

        MODEL_MANAGER.delete(self.name(), key, self.serializer_backend())

        app.logger.info(f'Successfully deleted model with key: {key}')

        return {'message': 'SUCCESS'}