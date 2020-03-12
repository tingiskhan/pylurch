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
put_parser.add_argument('name', type=str, help='Name of the series, used for persisting')
put_parser.add_argument('modkwargs', type=dict, help='Kwargs for model')
put_parser.add_argument('algkwargs', type=dict, help='Kwargs for algorithm')
put_parser.add_argument('retrain', type=str, help='Whether to retrain data', default='False')

get_parser = BASE_REQ.copy()
get_parser.add_argument('model-key', type=str, required=True, help='Key of the model')

patch_parser.add_argument('model-key', type=str, required=True, help='Key of the model')

post_parser = patch_parser.copy()
post_parser.add_argument('model-key', type=str, required=True, help='Key of the model')


# TODO: Add cache or something for storing models
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
        Returns the backend used by the backend
        :rtype: str
        """

        raise NotImplementedError()

    def done_callback(self, fut, key, x):
        """
        What to do when the callback is done.
        :param fut: The future
        :type fut: flask_executor.futures.Future
        :param key: The key associated with the model
        :type key: str
        """
        res = fut.result()
        app.logger.info(f'Successfully trained {key}, now trying to persist')

        try:
            bytestring = self.serialize(res, x)
            self.save_model(MODEL_MANAGER, key, bytestring)

            app.logger.info(f'Successfully persisted {key}')
        except Exception as e:
            app.logger.exception(f'Failed persisting {key}', e)
        finally:
            executor.futures.pop(key)

    def serialize(self, model, x, y=None):
        """
        Convert model to ONNX.
        :param model: The model to convert
        :param x: The data used for training
        :param y: The data used for training
        :return: onnx
        """

        raise NotImplementedError()

    def save_model(self, model_manager, key, mod):
        """
        Method for saving the model.
        :param model_manager: The model manager
        :type model_manager: ModelManager
        :param key: The key
        :type key: str
        :param mod: The model
        :type mod: bytes
        :return: Nothing
        :rtype: None
        """

        model_manager.save(key, mod)

    def load_model(self, model_manager, key):
        """
        Method for loading the model.
        :param model_manager: The model manager
        :type model_manager: ModelManager
        :param key: The key
        :type key: str
        :return: Model
        """

        return model_manager.load(key)

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

    def predict(self, mod, x, **kwargs):
        """
        Return the prediction.
        :param mod: The model
        :param x: The data to predict for
        :type x: pd.DataFrame
        :return: JSON like dict
        :rtype: dict
        """

        if self.serializer_backend() != SerializerBackend:
            raise NotImplementedError(f'You must override the method yourself!')

        inp_name = mod.get_inputs()[0].name
        label_name = mod.get_outputs()[0].name

        return {'y': mod.run([label_name], {inp_name: x.values.astype(np.float32)})[0].tolist()}

    def parse_data(self, data, **kwargs):
        """
        Method for parsing data.
        :param data: The data in string format
        :type data: str
        :return: Data in required format
        """

        return pd.read_json(data, **kwargs).sort_index()

    def check_model_status(self, key):
        """
        Helper function for checking the status of the model.
        :param key: The key of the model
        :type key: str
        :return: String indicating status
        :rtype: str
        """

        running_in_executor = executor.futures._state(key)

        if running_in_executor:
            return running_in_executor

        return MODEL_MANAGER.check_status(key)

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
        algkwargs = args['algkwargs'] or dict()

        if 'y' in args:
            algkwargs['y'] = self.parse_data(args['y'])

        model = self.make_model(**modkwargs)

        # ===== Generate model key ===== #
        data_key = args['name'] or hash_series(x)
        key = sha256((data_key + model.__class__.__name__).encode()).hexdigest()

        # ===== Check status ===== #
        status = self.check_model_status(key)

        if status == ModelStatus.Running:
            if retrain:
                return {
                    'message': f'Cannot cancel already {status} task. Try re-running when model is done',
                    'model-key': key
                }

            return {'message': status, 'model-key': key}

        if status is not None and not retrain:
            return {'message': status, 'model-key': key}

        futures = executor.submit_stored(key, run_model, self.fit, model, x, MODEL_MANAGER, key, **algkwargs)
        futures.add_done_callback(lambda u: self.done_callback(u, key, x))

        app.logger.info(f'Successfully started training of {model.__class__.__name__} using {x.shape[0]} observations')

        return {'model-key': key}

    @custom_login(auth_token.login_required)
    @custom_error
    def post(self):
        args = post_parser.parse_args()
        key = args['model-key']

        status = self.check_model_status(key)

        if status is None:
            return {'message': 'Model does not exist!'}, 400

        if status != ModelStatus.Done:
            return {'message': status}

        task = executor.futures.pop(key)

        if task is not None:
            mod = task.result()
        else:
            mod = self.load_model(MODEL_MANAGER, key)

        app.logger.info(f'Predicting values using model {mod.__class__.__name__}')

        return self.predict(mod, pd.read_json(args['x'], orient=args['orient']))

    @custom_login(auth_token.login_required)
    @custom_error
    def get(self):
        args = get_parser.parse_args()
        key = args['model-key']

        status = self.check_model_status(key)

        if status != ModelStatus.Done:
            return {'message': status}

        mod = self.load_model(MODEL_MANAGER, key)
        return self.get_return({'message': 'DONE'}, mod)

    @custom_login(auth_token.login_required)
    @custom_error
    def patch(self):
        args = patch_parser.parse_args()
        key = args['model-key']

        status = self.check_model_status(key)

        if status is None:
            return {'message': 'Model does not exist!'}, 400

        if status != ModelStatus.Done:
            return {'message': status}

        model = self.load_model(MODEL_MANAGER, key)
        x = self.parse_data(args['x'])

        kwargs = dict()
        if 'y' in args:
            kwargs['y'] = self.parse_data(args['y'])

        futures = executor.submit_stored(key, run_model, self.update, model, x, MODEL_MANAGER, key, **kwargs)
        futures.add_done_callback(lambda u: self.done_callback(u, key, x))

        app.logger.info(f'Started updating of model {model.__class__.__name__} using {x.shape[0]} new observations')

        return {'message': executor.futures._state(key)}

    @custom_login(auth_token.login_required)
    @custom_error
    def delete(self):
        args = get_parser.parse_args()
        key = args['model-key']

        app.logger.info(f'Deleting model with key: {key}')

        MODEL_MANAGER.delete(key)

        app.logger.info(f'Successfully deleted model with key: {key}')

        return {'message': 'SUCCESS'}