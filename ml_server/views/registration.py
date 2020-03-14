from flask_restful import Resource
from ..utils import BASE_REQ
from ..db.models import User
from ..app import SESSION, auth_basic
from flask import request

user_parser = BASE_REQ.copy()
user_parser.add_argument('username', type=str, required=True, help='User name')

parser = user_parser.copy()
parser.add_argument('password', type=str, required=True, help='Password')

alter_parser = user_parser.copy()
alter_parser.add_argument('admin', type=int)


class RegistrationView(Resource):
    def post(self):
        args = parser.parse_args()
        username = args['username']

        session = SESSION()

        if session.query(User).filter(User.username == username).count() > 0:
            session.close()
            return {'username': username, 'status': 'exists'}

        user = User(username, args['password'])

        session.add(user)
        session.commit()
        session.close()

        return {'username': username, 'status': 'created'}

    @auth_basic.login_required
    def get(self):
        session = SESSION()
        user = session.query(User).filter(User.username == request.authorization['username']).one()

        if not user.admin:
            session.close()
            return 403

        users = session.query(User).all()
        usernames = [u.username for u in users]
        session.close()

        return {'users': usernames}

    @auth_basic.login_required
    def put(self):
        session = SESSION()
        user = session.query(User).filter(User.username == request.authorization['username']).one()

        if not user.admin and session.query(User).count() > 1:
            session.close()
            return 403

        args = alter_parser.parse_args()

        if args['username'] == user.username:
            to_change = user
        else:
            to_change = session.query(User).filter(User.username == args['username']).one()

        to_change.admin = bool(args['admin'])

        session.commit()
        session.close()

        return {'status': 'success'}

    @auth_basic.login_required
    def delete(self):
        session = SESSION()
        user = session.query(User).filter(User.username == request.authorization['username']).one()

        if not user.admin:
            session.close()
            return 403

        args = user_parser.parse_args()
        to_delete = session.query(User).filter(User.username == args['username']).one()
        session.delete(to_delete)

        session.commit()
        session.close()

        return {'status': 'success'}


class LoginView(Resource):
    @auth_basic.login_required
    def get(self):
        session = SESSION()
        user = session.query(User).filter(User.username == request.authorization['username']).one()
        auth_token = user.generate_auth_token().decode()

        session.close()

        return {'token': auth_token}
