from typing import Union, List
from ...database import Artifact
from .base import Context


class PredictionContext(Context):
    def on_exit(self):
        return

    def _get_result(self, session_id: int) -> Union[Artifact, List[Artifact]]:
        return self._client.get(Artifact, lambda u: u.session_id == session_id)

    def get_result(self):
        return self._get_result(self._session.id)
