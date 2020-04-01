from flask_restful import Resource
from ml_api.utils import BASE_REQ
from ..db.models import User
from ..app import db, auth_basic
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

        if db.session.query(User).filter(User.username == username).count() > 0:
            return {'username': username, 'status': 'exists'}

        user = User(username, args['password'])

        db.session.add(user)
        db.session.commit()

        return {'username': username, 'status': 'created'}

    @auth_basic.login_required
    def get(self):
        user = db.session.query(User).filter(User.username == request.authorization['username']).one()

        if not user.admin:
            db.session.close()
            return 403

        users = db.session.query(User).all()
        usernames = [u.username for u in users]
        
        return {'users': usernames}

    @auth_basic.login_required
    def put(self):
        user = db.session.query(User).filter(User.username == request.authorization['username']).one()

        if not user.admin and db.session.query(User).count() > 1:
            db.session.close()
            return 403

        args = alter_parser.parse_args()

        if args['username'] == user.username:
            to_change = user
        else:
            to_change = db.session.query(User).filter(User.username == args['username']).one()

        to_change.admin = bool(args['admin'])

        db.session.commit()

        return {'status': 'success'}

    @auth_basic.login_required
    def delete(self):
        user = db.session.query(User).filter(User.username == request.authorization['username']).one()

        if not user.admin:
            db.session.close()
            return 403

        args = user_parser.parse_args()
        to_delete = db.session.query(User).filter(User.username == args['username']).one()
        db.session.delete(to_delete)

        db.session.commit()

        return {'status': 'success'}


class LoginView(Resource):
    @auth_basic.login_required
    def get(self):
        user = db.session.query(User).filter(User.username == request.authorization['username']).one()
        auth_token = user.generate_auth_token().decode()

        return {'token': auth_token}
