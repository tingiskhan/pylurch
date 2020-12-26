from typing import Union
from pyalfred.contract.interface import DatabaseInterface
from ..database import TrainingSession, Model, BaseMixin
from .context import ClientTrainingContext, ClientPredictionContext


class SessionInterface(DatabaseInterface):
    def __init__(self, base_url: str):
        super().__init__(base_url, mixin_ignore=BaseMixin)
        self._client = DatabaseInterface(base_url)

    def _get_session(self, model: Model, session_name: str, only_succeeded=False, latest=True):
        def f(u: TrainingSession):
            if only_succeeded:
                return (u.name == session_name) & (u.model_id == model.id) & (u.has_model == True)

            return (u.name == session_name) & (u.model_id == model.id)

        return self.get(TrainingSession, f, latest=latest)

    def _create_session(self, model: Model, session_name: str) -> TrainingSession:
        latest_session = self._get_session(model, session_name)

        session = TrainingSession(
            model_id=model.id, name=session_name, version=1 if latest_session is None else (latest_session.version + 1),
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
        session = self._create_session(model, session_name)

        return ClientTrainingContext(self, session)

    def update_session(self, *args, **kwargs):
        raise NotImplementedError()

    def begin_prediction_session(self, session_id: int):
        session = self._client.get(TrainingSession, lambda u: u.id == session_id, one=True)

        if session is None:
            raise ValueError(f"No {TrainingSession.__name__} exists with id: {session_id}!")

        return ClientPredictionContext(self._client, session)

    def get_session(
        self, model_name: str, model_revision: str, session_name: str, only_succeeded=False
    ) -> Union[TrainingSession, None]:

        model = self._get_model(model_name, model_revision)
        return self._get_session(model, session_name, only_succeeded=only_succeeded)
