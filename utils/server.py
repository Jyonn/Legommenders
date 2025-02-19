import os

import requests
from pigmento import pnt

from utils.config_init import AuthInit


class BaseResp:
    def __init__(self, resp: dict):
        self.msg = resp.get('msg', None)
        self.identifier = resp.get('identifier', None)
        self.append_msg = resp.get('append_msg', None)
        self.debug_msg = resp.get('debug_msg', None)
        self.code = resp.get('code', None)
        self.body = resp.get('body', None)
        self.http_code = resp.get('http_code', None)

    @property
    def ok(self):
        return self.identifier == 'OK'


class ExperimentBody:
    def __init__(self, body: dict):
        self.signature = body.get('signature', None)
        self.seed = body.get('seed', None)
        self.session = body.get('session', None)
        self.log = body.get('log', None)
        self.performance = body.get('performance', None)
        self.is_completed = body.get('is_completed', None)
        self.created_at = body.get('created_at', None)
        self.pid = body.get('pid', None)


class EvaluationBody:
    def __init__(self, body: dict):
        self.signature = body.get('signature', None)
        self.command = body.get('command', None)
        self.configuration = body.get('configuration', None)
        self.created_at = body.get('created_at', None)
        self.modified_at = body.get('modified_at', None)
        self.comment = body.get('comment', None)
        self.experiments = body.get('experiments', [])
        for i, experiment in enumerate(self.experiments):
            self.experiments[i] = ExperimentBody(experiment)


class Server:
    def __init__(self, uri, auth):
        self.uri = uri
        self.auth = auth
        self.pid = os.getpid()

    @classmethod
    def auto_auth(cls):
        uri, auth = AuthInit.get('lego_uri'), AuthInit.get('lego_auth')
        return cls(uri=uri, auth=auth)

    @staticmethod
    def calculate_bytes(data: dict):
        return sum(len(key) + len(str(value)) for key, value in data.items())

    def post(self, uri, data):
        total_bytes = self.calculate_bytes(data)
        pnt(f'Uploading {total_bytes} bytes of data to {uri}')

        with requests.post(
            uri,
            headers={'Authentication': self.auth},
            json=data,
        ) as response:
            return BaseResp(response.json())

    def put(self, uri, data):
        total_bytes = self.calculate_bytes(data)
        pnt(f'Uploading {total_bytes} bytes of data to {uri}')

        with requests.put(
            uri,
            headers={'Authentication': self.auth},
            json=data,
        ) as response:
            return BaseResp(response.json())

    def get(self, uri, query):
        pnt(f'Sending query request to {uri}')

        with requests.get(
            uri,
            headers={'Authentication': self.auth},
            params=query
        ) as response:
            return BaseResp(response.json())

    def get_experiment_info(self, session):
        query = {
            'session': session
        }
        return self.get(f'{self.uri}/experiments/', query)

    def create_or_get_evaluation(self, signature, command, configuration):
        data = {
            'signature': signature,
            'command': command,
            'configuration': configuration,
        }
        return self.post(f'{self.uri}/evaluations/', data)

    def create_or_get_experiment(self, signature, seed):
        data = {
            'signature': signature,
            'seed': seed,
        }
        return self.post(f'{self.uri}/experiments/', data)

    def register_experiment(self, session):
        data = {
            'pid': self.pid,
        }
        return self.post(f'{self.uri}/experiments/{session}/register', data)

    def complete_experiment(self, session, log, performance):
        data = {
            'session': session,
            'log': log,
            'performance': performance,
        }
        return self.put(f'{self.uri}/experiments/', data)


if __name__ == '__main__':
    server = Server(
        uri=AuthInit.get('uri'),
        auth=AuthInit.get('auth'),
    )

    evaluation = server.create_or_get_evaluation('123456', 'command', '{"config": "data"}')
