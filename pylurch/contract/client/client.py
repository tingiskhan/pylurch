from typing import Union
from pyalfred.contract.client import Client
from ..database import TrainingSession, Model, BaseMixin
from .context import TrainingContext, PredictionContext, UpdateContext


class InferenceClient(Client):
    def __init__(self, base_url: str):
        super().__init__(base_url, mixin_ignore=BaseMixin)

    def _get_training_session(self, model_id: int, session_name: str, only_succeeded=False, latest=True):
        def f(u: TrainingSession):
            if only_succeeded:
                return (u.name == session_name) & (u.model_id == model_id) & (u.has_model == True)

            return (u.name == session_name) & (u.model_id == model_id)

        if latest:
            return self.get(TrainingSession, f, one=True, operations="order by id desc,first")

        return self.get(TrainingSession, f)

    def _create_training_session(self, model_id: int, session_name: str) -> TrainingSession:
        latest_session = self._get_training_session(model_id, session_name)

        session = TrainingSession(
            model_id=model_id, name=session_name, version=1 if latest_session is None else (latest_session.version + 1),
        )

        return self.create(session)

    def _get_model(self, name: str, revision: str) -> Model:
        return self.get(Model, lambda u: (u.name == name) & (u.revision == revision), one=True)

    def _get_create_model(self, name: str, revision: str) -> Model:
        model = self._get_model(name, revision)

        if model is not None:
            return model

        return self.create(Model(name=name, revision=revision))

    def begin_training_session(self, model_name: str, model_revision: str, session_name: str):
        model = self._get_create_model(model_name, model_revision)
        session = self._create_training_session(model.id, session_name)

        return TrainingContext(self, session)

    def begin_update_session(self, new_session_name: str, old_training_session: int):
        old_session = self.get(TrainingSession, lambda u: u.id == old_training_session, one=True)

        if (old_session is None) or not old_session.has_model:
            raise ValueError(f"The TrainingSession with id {old_training_session} does not exist!")

        new_session = self._create_training_session(old_session.model_id, new_session_name)

        return UpdateContext(self, new_session, old_session)

    def begin_prediction_session(self, session_id: int):
        session = self.get(TrainingSession, lambda u: u.id == session_id, one=True)

        if session is None:
            raise ValueError(f"No {TrainingSession.__name__} exists with id: {session_id}!")

        return PredictionContext(self, session)

    def get_session(
        self, model_name: str, model_revision: str, session_name: str, only_succeeded=False
    ) -> Union[TrainingSession, None]:

        model = self._get_model(model_name, model_revision)
        return self._get_training_session(model.id, session_name, only_succeeded=only_succeeded)
