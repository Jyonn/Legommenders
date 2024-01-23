from typing import Dict

from torch import nn


class BaseModule(nn.Module):
    def request(self) -> Dict[str, list]:
        pass

    def response(self, requester_name: str, request: list) -> Dict[str, any]:
        response = {}
        for attr in request:
            if attr == 'self':
                response[attr] = self
            else:
                response[attr] = getattr(self, attr, None)
        return response

    def receive(self, responser_name: str, response: dict):
        pass
