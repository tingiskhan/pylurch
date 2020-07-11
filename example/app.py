from falcon import API as Api
from ml_api.server.resources import ModelResource, DatabaseResource
from ml_api.contract.database import Base
from ml_api.contract.interfaces import LocalDatabaseInterface
from concurrent.futures import ThreadPoolExecutor as Executor
from ml_api.utils import make_base_logger
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import create_engine
import os
from inspect import isclass

# ===== Database related ===== #
sqlite = 'sqlite:///debug-database.db?check_same_thread=false'

engine = create_engine(
    os.environ.get('SQLALCHEMY_DATABASE_URI', sqlite),
    **os.environ.get('SQLALCHEMY_ENGINE_OPTIONS', {'pool_pre_ping': True})
)

Session = scoped_session(sessionmaker(bind=engine))

# ===== Executor for running background processes ===== #
executor = Executor()


def queue_method(obj, key, func, *args, **kwargs):
    futures = executor.submit(key, func, *args, **kwargs)

    return


# ===== Initialization script ===== #
def init_app():
    api = Api()

    # ===== Initialize everything ===== #
    Base.metadata.create_all(bind=engine)

    # ===== Add database interface ===== #
    import ml_api.contract.schemas as schemas
    objs = [
        s for k, s in vars(schemas).items() if
        isclass(s) and issubclass(s, schemas.BaseSchema) and not k.startswith('Base')
    ]

    for o in objs:
        api.add_route(f'/{o.endpoint()}', DatabaseResource(o, Session))

    # ===== Add models ===== #
    from .models import LinearRegressionModel, LogisticRegressionModel, NeuralNetworkModel

    intf = LocalDatabaseInterface(Session)

    api.add_route(
        '/linreg',
        ModelResource(LinearRegressionModel(intf), queue_method)
    )

    api.add_route(
        '/logreg',
        ModelResource(LogisticRegressionModel(intf), queue_method)
    )

    api.add_route(
        '/nn',
        ModelResource(NeuralNetworkModel(intf), queue_method)
    )

    logger = make_base_logger(__name__)
    logger.info('Successfully registered all views')

    return api