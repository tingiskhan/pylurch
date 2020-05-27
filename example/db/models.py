from ..app import bcrypt, db, app
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer, BadSignature, SignatureExpired
from sqlalchemy import Column, String, Boolean
from ml_api.database import BaseMixin, Base


class User(BaseMixin, Base):
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
        """

        s = Serializer(app.config['SECRET_KEY'], expires_in=app.config.get('TOKEN_EXPIRATION', 12 * 60 * 60))
        return s.dumps({'id': self.id})

    def verify_password(self, password) -> bool:
        """
        Verifies the hashed password.
        :param password: Password
        :type password: str
        """

        return bcrypt.check_password_hash(self.password, password)

    @staticmethod
    def verify_auth_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except SignatureExpired:
            return None
        except BadSignature:
            return None

        is_ok = db.session.query(User).get(data['id']) is not None

        return is_ok