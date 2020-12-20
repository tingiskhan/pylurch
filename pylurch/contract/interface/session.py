from typing import Union
from pyalfred.contract.interface import DatabaseInterface
from ..database import TrainingSession, Model, BaseMixin
from .training_context import TrainingContext


class SessionInterface(DatabaseInterface):
    def __init__(self, base_url: str):
        super().__init__(base_url, mixin_ignore=BaseMixin)
        self._intf = DatabaseInterface(base_url)

    def _get_session(self, model: Model, session_name: str, only_succeeded=False):
        def f(u: TrainingSession):
            if only_succeeded:
                return (u.name == session_name) & (u.model_id == model.id) & (u.has_result == "True")

            return (u.name == session_name) & (u.model_id == model.id)

        return self.get(TrainingSession, f, latest=True)

    def _create_session(self, model: Model, session_name: str):
        latest_session = self._get_session(model, session_name)

        session = TrainingSession(
            model_id=model.id, name=session_name, version=1 if latest_session is None else (latest_session.version + 1),
        )

        return self.create(session)

    def _get_model(self, model_name: str) -> Model:
        return self.get(Model, lambda u: u.name == model_name, one=True)

    def _get_create_model(self, model_name: str):
        model = self._get_model(model_name)

        if model is not None:
            return model

        return self.create(Model(name=model_name))

    def begin_session(self, model_name: str, session_name: str) -> TrainingContext:
        model = self._get_create_model(model_name)
        session = self._create_session(model, session_name)

        return TrainingContext(self, session)

    def update_session(self, session_id: int) -> TrainingContext:
        session = self.get(TrainingSession, lambda u: u.id == session_id, one=True)

        if session is None:
            raise ValueError(f"No {TrainingSession.__class__.__name__} exists with id: {session_id}!")

        return TrainingContext(self, session)

    def get_session(self, model_name: str, session_name: str, only_succeeded=False) -> Union[TrainingSession, None]:
        model = self._get_model(model_name)
        return self._get_session(model, session_name, only_succeeded=only_succeeded)
