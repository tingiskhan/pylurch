from pylurch.contract.client import SessionClient
from .blueprint import InferenceModelBlueprint, TModel, TOutput
from .session import TrainingSession, PredictionSession, UpdateSession


# TODO: Should this be context as well...?
class SessionManager(object):
    def __init__(self, client: SessionClient, blueprint: InferenceModelBlueprint[TModel, TOutput]):
        self._client = client
        self._blueprint = blueprint

    def begin_training_session(self, session_name: str, **kwargs):
        context = self._client.begin_training_session(
            self._blueprint.name(), self._blueprint.get_revision(), session_name
        )

        return TrainingSession(context, self._blueprint, **kwargs)

    def begin_prediction_session(self, session_id: int):
        context = self._client.begin_prediction_session(session_id)

        return PredictionSession(context, self._blueprint)

    def begin_update_session(self, new_session_name: str, old_session_id: int):
        context = self._client.begin_update_session(new_session_name, old_session_id)

        return UpdateSession(context, self._blueprint)
