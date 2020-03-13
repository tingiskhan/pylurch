from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy import Column, String, Boolean, DateTime, func, LargeBinary, Integer, ForeignKey
from sqlalchemy.orm import relationship
from ml_server.app import bcrypt, SESSION, app
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer, BadSignature, SignatureExpired
from .enums import ModelStatus, SerializerBackend
from datetime import datetime
import platform
import onnxruntime as rt
import dill


class MyMixin(object):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__

    __mapper_args__ = {
        'always_refresh': True
    }

    id = Column(Integer, primary_key=True)
    upd_at = Column(DateTime, nullable=True, default=func.now())
    upd_by = Column(String(255), nullable=False, default=platform.node(), onupdate=platform.node())
    last_update = Column(DateTime, server_default=func.now(), onupdate=func.now())


Base = declarative_base()


class User(MyMixin, Base):
    username = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    admin = Column(Boolean, nullable=False, default=False)

    def __init__(self, username, password, admin=False):
        """
        Base class for defining a user.
        :param username: The user name
        :type username: str
        :param password: The password
        :type password: str
        :param admin: Whether admin or not
        :type admin: bool
        """

        self.username = username
        self.password = bcrypt.generate_password_hash(password).decode()
        self.admin = admin

    # Below, see: https://blog.miguelgrinberg.com/post/restful-authentication-with-flask
    def generate_auth_token(self):
        """
        Generates a token.
        :rtype: basestring
        """

        s = Serializer(app.config['SECRET-KEY'], expires_in=app.config.get('TOKEN-EXPIRATION'))
        return s.dumps({'id': self.id})

    def verify_password(self, password):
        """
        Verifies the hashed password.
        :param password: Password
        :type password: str
        :return: Whether correct
        :rtype: bool
        """

        return bcrypt.check_password_hash(self.password, password)

    @staticmethod
    def verify_auth_token(token):
        s = Serializer(app.config['SECRET-KEY'])
        try:
            data = s.loads(token)
        except SignatureExpired:
            return None
        except BadSignature:
            return None

        session = SESSION()
        is_ok = session.query(User).get(data['id']) is not None
        session.close()

        return is_ok


class Model(MyMixin, Base):
    name = Column(String(255), nullable=False, unique=True)

    training_sessions = relationship('TrainingSession', back_populates='model')

    def __init__(self, name):
        self.name = name


class TrainingSession(MyMixin, Base):
    model_id = Column(Integer, ForeignKey(Model.id))
    hash_key = Column(String(255), nullable=False)

    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False, default=func.now())

    status = Column(String(255), nullable=False)

    backend = Column(String(255), nullable=False)
    byte_string = Column(LargeBinary(), nullable=True)

    model = relationship(Model, back_populates='training_sessions', use_list=False)

    def __init__(self, hash_key, start_time, status, backend, end_time=datetime.max, byte_string=None):
        if status not in ModelStatus():
            raise NotImplementedError(f'status must be in: {ModelStatus()}')
        if backend not in SerializerBackend():
            raise NotImplementedError(f'Status must be in: {SerializerBackend()}')

        self.hash_key = hash_key
        self.start_time = start_time
        self.status = status
        self.backend = backend
        self.end_time = end_time
        self.byte_string = byte_string

    def load(self):
        if self.backend == SerializerBackend.ONNX:
            return rt.InferenceSession(self.byte_string)
        elif self.backend == SerializerBackend.Dill:
            return dill.loads(self.byte_string)