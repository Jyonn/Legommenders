import requests


class Server:
    def __init__(self, uri, auth):
        self.uri = uri
        self.auth = auth

    def calculate_bytes(self, data: dict):
        return sum(len(key) + len(str(value)) for key, value in data.items())

    def post(self, signature, command, log, config, performance):
        data = {
            'signature': signature,
            'command': command,
            'configuration': config,
            'log': log,
            'performance': performance,
        }
        total_bytes = self.calculate_bytes(data)

        print(f'Uploading {total_bytes} bytes of data')

        """Creates a new evaluation entry."""
        return requests.post(
            f'{self.uri}/evaluations/',
            headers={'Authentication': self.auth},
            json=data,
        )
