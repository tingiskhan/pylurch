from flask import Flask
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
import os
from flask_httpauth import HTTPTokenAuth, HTTPBasicAuth
import logging
from ml_api import MachineLearningApi
from ml_api.model_managers import SQLModelManager


# ===== Setup extensions ===== #
app = Flask(__name__)

# ===== Read config ====== #
if 'ML_API_CONFIG' in os.environ:
    app.config.from_envvar('ML_API_CONFIG')

sqlite = 'sqlite:///debug-database.db?check_same_thread=false'

if not app.config.get('SECRET_KEY'):
    app.config['SECRET_KEY'] = os.urandom(24)

app.config.setdefault('SQLALCHEMY_DATABASE_URI', os.environ.get('DB_CONNSTRING') or sqlite)
app.config.setdefault('EXECUTOR_PROPAGATE_EXCEPTIONS', True)
app.config.setdefault('PRODUCTION', os.environ.get('PRODUCTION') or False)
app.config.setdefault('SQLALCHEMY_TRACK_MODIFICATIONS', False)

bcrypt = Bcrypt(app)
db = SQLAlchemy(app)

# ===== Model manager =====#
model_manager = SQLModelManager(db.session)
api = MachineLearningApi(app=app, model_manager=model_manager)

# ===== Auth ====== #
auth_token = HTTPTokenAuth(scheme='Token')
auth_basic = HTTPBasicAuth()
admin_auth = HTTPBasicAuth()


def init_app():
    # ===== Initialize everything ===== #
    model_manager.set_logger(app.logger)

    # ===== Logging ===== #
    app.logger.setLevel(logging.DEBUG)

    # ===== Define tables ====== #
    from .db.models import Base, User

    app.logger.info(f'Trying to connect to {repr(db.session.bind.url)}')

    Base.metadata.create_all(db.session.bind)

    app.logger.info('Successfully connected to database and created relevant tables')

    # ===== Create a root user ===== #
    session = db.session

    if not session.query(User).filter(User.username == 'admin').one_or_none():
        admin = User('admin', os.environ.get('ML_ADMIN_PW', 'this-is-not-a-good-password'), admin=True)
        session.add(admin)
        session.commit()

        app.logger.info('Successfully created an admin user')

    app.logger.info(f'Application is configured as: {"production" if app.config["PRODUCTION"] else "debug"}')

    # ===== Authorization logic ===== #
    @auth_basic.verify_password
    def verify(username, password):
        user = session.query(User).filter(User.username == username).one()
        verified = user.verify_password(password)

        return verified

    @auth_token.verify_token
    def verify_token(token):
        return User.verify_auth_token(token)

    # TODO: Duplication
    @admin_auth.verify_password
    def verify(username, password):
        user = session.query(User).filter(User.username == username).one()
        verified = user.verify_password(password) and user.admin

        return verified

    # ===== Define views ===== #
    from .views.example import HelloWorld
    from .views.admin import AdminView
    from .views.regression import LinearRegressionView, LogisticRegressionView

    api.add_resource(HelloWorld, '/hello-world')
    api.add_resource(AdminView, '/admin')

    api.add_resource(LinearRegressionView, '/linreg')
    api.add_resource(LogisticRegressionView, '/logreg')

    from .views.registration import LoginView, RegistrationView

    api.add_resource(LoginView, '/login')
    api.add_resource(RegistrationView, '/register')

    app.logger.info('Registering login views')

    app.logger.info('Finished setting up application')

    return app