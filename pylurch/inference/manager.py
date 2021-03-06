from pylurch.contract.client import InferenceClient
from .blueprint import InferenceModelBlueprint


# TODO: Should this be context as well...?
class InferenceManager(object):
    def __init__(self, client: InferenceClient):
        self._client = client

    def begin_training_session(self, blueprint: InferenceModelBlueprint, session_name: str):
        return self._client.begin_training_session(
            blueprint.name(), blueprint.get_revision(), session_name
        )

    def begin_prediction_session(self, session_id: int):
        return self._client.begin_prediction_session(session_id)

    def begin_update_session(self, new_session_name: str, old_session_id: int):
        return self._client.begin_update_session(new_session_name, old_session_id)
