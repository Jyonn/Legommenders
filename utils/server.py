import os.path

import requests


class Server:
    def __init__(self, uri, auth):
        self.uri = uri
        self.auth = auth

    def read(self, path):
        with open(path, 'r') as f:
            return f.read()

    def calculate_bytes(self, data: dict):
        return sum(len(key) + len(str(value)) for key, value in data.items())

    def post(self, signature, command, log, base_dir):
        """
        [00:00:00] |Trainer| SIGNATURE: EDWmZVNS
        [00:00:00] |Trainer| BASE DIR: checkpoints/mind/NRMS
        """
        _, data_name, model_name = base_dir.split('/')
        config_path = os.path.join(base_dir, f'{signature}.json')
        performance_path = os.path.join(base_dir, f'{signature}.csv')

        data = {
            'signature': signature,
            'command': command,
            'configuration': self.read(config_path),
            'log': log,
            'performance': self.read(performance_path),
        }
        total_bytes = self.calculate_bytes(data)

        print(f'Uploading {total_bytes} bytes of data')

        """Creates a new evaluation entry."""
        return requests.post(
            f'{self.uri}/evaluations',
            headers={'Authorization': self.auth},
            json=data,
        )
