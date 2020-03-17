from flask import Flask
from flask_restful import Api
from flask_executor import Executor
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
import os
from flask_httpauth import HTTPTokenAuth, HTTPBasicAuth
import logging

# ===== Setup service ===== #
app = Flask(__name__)

# ===== Setup extensions ===== #
api = Api()
bcrypt = Bcrypt()
executor = Executor()
db = SQLAlchemy()

# ===== Model manager =====#
from .model_managers import SQLModelManager

MODEL_MANAGER = SQLModelManager(app.logger, db.session)

# ===== Auth ====== #
auth_token = HTTPTokenAuth(scheme='Token')
auth_basic = HTTPBasicAuth()
admin_auth = HTTPBasicAuth()


def init_app(ignore_models=False):
    # ===== Read config ====== #
    if 'ML_API_CONFIG' in os.environ:
        app.config.from_envvar('ML_API_CONFIG')

    sqlite = 'sqlite:///debug-database.db?check_same_thread=false'

    if not app.config.get('SECRET_KEY'):
        app.config['SECRET_KEY'] = os.urandom(24)

    app.config.setdefault('SQLALCHEMY_DATABASE_URI', os.environ.get('DB_CONNSTRING') or sqlite)
    app.config.setdefault('TOKEN_EXPIRATION', 12 * 10 * 60)
    app.config.setdefault('EXECUTOR_PROPAGATE_EXCEPTIONS', True)
    app.config.setdefault('PRODUCTION', os.environ.get('PRODUCTION') or False)
    app.config.setdefault('EXTENSION', 'pkl')
    app.config.setdefault('EXTERNAL_AUTH', os.environ.get('EXTERNAL_AUTH') or True)
    app.config.setdefault('SQLALCHEMY_TRACK_MODIFICATIONS', False)

    # ===== Initialize everything ===== #
    api.__init__(app)
    bcrypt.__init__(app)
    executor.__init__(app)
    db.__init__(app)

    # ===== Logging ===== #
    app.logger.setLevel(logging.DEBUG)

    # ===== Define tables ====== #
    from .db.models import Base, User

    Base.metadata.create_all(db.session.bind)

    app.logger.info('Successfully connected to database and created relevant tables')

    # ===== Create a root user ===== #
    session = db.session

    if not session.query(User).filter(User.username == 'admin').one_or_none():
        admin = User('admin', os.environ.get('ML_ADMIN_PW', 'this-is-not-a-good-password'), admin=True)
        session.add(admin)
        session.commit()

        app.logger.info('Successfully created an admin user')

    # ===== Define model manager ====== #
    MODEL_MANAGER.close_all_running()
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

    if not ignore_models:
        api.add_resource(LinearRegressionView, '/linreg')
        api.add_resource(LogisticRegressionView, '/logreg')

    if not app.config['EXTERNAL_AUTH']:
        from .views.registration import LoginView, RegistrationView

        api.add_resource(LoginView, '/login')
        api.add_resource(RegistrationView, '/register')

        app.logger.info('Registering login views')
    else:
        app.logger.info('Using external authentication')

    app.logger.info('Finished setting up application')

    return app