from falcon import API as Api
from ml_api import ModelResource
from ml_api.model_managers import SQLModelManager
from concurrent.futures import ThreadPoolExecutor as Executor
from logging import DEBUG, getLogger, StreamHandler, Formatter
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine
import os

# ===== Model manager =====#
sqlite = 'sqlite:///debug-database.db?check_same_thread=false'

engine = create_engine(
    os.environ.get('SQLALCHEMY_DATABASE_URI', sqlite),
    **os.environ.get('SQLALCHEMY_ENGINE_OPTIONS', {'pool_pre_ping': True})
)

Session = scoped_session(sessionmaker(bind=engine))

model_manager = SQLModelManager(Session)
model_manager.initialize()
executor = Executor()


def queue_method(obj, key, func, *args, **kwargs):
    futures = executor.submit(key, func, *args, **kwargs)

    return


logger = getLogger(__name__)

ch = StreamHandler()
ch.setLevel(DEBUG)

formatter = Formatter('[%(asctime)s] %(levelname)s in %(name)s: %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.setLevel(DEBUG)


def init_app():
    api = Api()

    # ===== Initialize everything ===== #
    model_manager.set_logger(logger)

    # ===== Add models ===== #
    from .models.regression import LinearRegressionModel, LogisticRegressionModel

    api.add_route(
        '/linreg', ModelResource(LinearRegressionModel('linear-regression', logger, model_manager), queue_method)
    )
    api.add_route(
        '/logreg', ModelResource(LogisticRegressionModel('logistic-regression', logger, model_manager), queue_method)
    )

    logger.info('Successfully registered all views')

    return api