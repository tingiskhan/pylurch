import os
from abc import ABC
from .base import BaseModelManager
import platform
from datetime import datetime
import glob
from ml_server.db.enums import ModelStatus, SerializerBackend
import yaml


class FileModelManager(BaseModelManager, ABC):
    def __init__(self, logger, prefix, **kwargs):
        """
        Defines a base class for model management.
        :param prefix: The prefix for each model file
        :type prefix: str
        """

        super().__init__(logger, **kwargs)

        self._pref = prefix

    def close_all_running(self):
        ymls = glob.glob(f'{self._pref}/*.yml', recursive=True)

        running = 0
        for f in ymls:
            yml = self._get_yml(f)

            if yml.get('platform') != platform.node():
                continue
            if yml['status'] != ModelStatus.Running:
                continue

            yml['end-time'] = datetime.now()
            yml['status'] = ModelStatus.Failed

            self._save_yml(f, yml)

            running += 1

        if running > 0:
            self._logger.info(f'Encountered {running} running training session, but just started - closing!')

        return

    @staticmethod
    def _get_ext(backend):
        return 'onnx' if backend == SerializerBackend.ONNX else 'pkl'

    def _format_name(self, name, key, backend, yml=False):
        first = f'{name}-{key}'

        if not yml:
            return f'{first}.{self._get_ext(backend)}'

        return f'{first}-{backend}.yml'

    def pre_model_start(self, name, key, backend):
        yml = self._make_yaml(name, key, backend)

        yml['start-time'] = datetime.now()
        yml['end-time'] = datetime.max
        yml['status'] = ModelStatus.Running

        yml_name = self._format_name(f'{self._pref}/{name}', key, backend, yml=True)
        self._save_yml(yml_name, yml)

        return self

    def model_fail(self, name, key, backend):
        yml_name = self._format_name(f'{self._pref}/{name}', key, backend, yml=True)
        yml = self._get_yml(yml_name)

        yml['end-time'] = datetime.now()
        yml['status'] = ModelStatus.Failed

        return self

    def save(self, name, key, obj, backend):
        # ===== YAML ===== #
        yml_name = self._format_name(f'{self._pref}/{name}', key, backend, yml=True)
        yml = self._get_yml(yml_name)

        yml['end-time'] = datetime.now()
        yml['status'] = ModelStatus.Done

        self._save_yml(yml_name, yml)

        # ====== Model ===== #
        name = f'{self._pref}/{self._format_name(name, key, backend)}'
        self._save(name, obj)

        return self

    def check_status(self, name, key, backend):
        yml_name = self._format_name(f'{self._pref}/{name}', key, backend, yml=True)
        yml = self._get_yml(yml_name)

        if yml is None:
            return None

        return yml['status']

    def delete(self, name, key, backend):
        raise NotImplementedError()

    def _get_yml(self, path):
        raise NotImplementedError()

    def _save_yml(self, path, yml):
        raise NotImplementedError()

    def _save(self, name, obj):
        raise NotImplementedError()

    def _make_yaml(self, name, key, backend):
        data = {
            'model-name': name,
            'hash-key': key,
            'backend': backend,
            'platform': platform.node()
        }

        return data


class LocalModelManager(FileModelManager):
    def __init__(self, *args, **kwargs):
        """
        Defines a model manager for use in debug settings.
        """

        super().__init__(*args, **kwargs)

        if not os.path.exists(self._pref):
            os.mkdir(self._pref)

    def _save_yml(self, path, yml):
        with open(path, 'w') as f:
            yaml.dump(yml, f)

    def _get_yml(self, path):
        with open(path, 'r') as s:
            return yaml.safe_load(s)

    def _save(self, name, obj):
        with open(name, 'wb') as f:
            f.write(obj)

        return

    def _load(self, name, key, backend):
        name = f'{self._pref}/{self._format_name(name, key, backend)}'

        if not os.path.exists(name):
            return None

        with open(name, 'rb') as f:
            return f.read()

    def delete(self, name, key, backend):
        f = glob.glob(f'{self._pref}/*{self._format_name(name, key, backend)}', recursive=True)

        if len(f) > 1:
            raise ValueError('Multiple models with same name!')

        os.remove(f[-1])

        return self