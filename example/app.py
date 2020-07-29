from falcon import API as Api
from pylurch.server.resources import ModelResource
from pylurch.contract.interfaces import DatabaseInterface
from pylurch.utils import make_base_logger
from pylurch.server.tasking import ExecutorWrapper
import os


# ===== Initialization script ===== #
def init_app():
    api = Api()

    # ===== Add models ===== #
    from .models import LinearRegressionModel, LogisticRegressionModel, NeuralNetworkModel

    intf = DatabaseInterface(os.environ.get('DATABASE_URI', 'http://localhost:8081'))
    manager = ExecutorWrapper(intf)

    api.add_route(
        '/linreg',
        ModelResource(LinearRegressionModel(intf), manager)
    )

    api.add_route(
        '/logreg',
        ModelResource(LogisticRegressionModel(intf), manager)
    )

    api.add_route(
        '/nn',
        ModelResource(NeuralNetworkModel(intf), manager)
    )

    logger = make_base_logger(__name__)
    logger.info('Successfully registered all views')

    return api