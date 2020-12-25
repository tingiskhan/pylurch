from pylurch.contract.interface import SessionInterface
from .blueprint import InferenceModelBlueprint, TModel, TOutput
from .context import TrainingSessionContext, PredictionSessionContext


# TODO: Should this be context as well...?
class SessionManager(object):
    def __init__(self, client: SessionInterface, blueprint: InferenceModelBlueprint[TModel, TOutput]):
        self._client = client
        self._blueprint = blueprint

    def begin_training_session(self, session_name: str, **kwargs):
        context = self._client.begin_training_session(
            self._blueprint.name(), self._blueprint.git_commit_hash(), session_name
        )

        return TrainingSessionContext(context, self._blueprint, **kwargs)

    def begin_prediction_session(self, session_id: int):
        context = self._client.begin_prediction_session(session_id)

        return PredictionSessionContext(context, self._blueprint)
