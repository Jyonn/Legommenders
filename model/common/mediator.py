from typing import cast, Dict

from pigmento import pnt


class ModuleType:
    legommender = 'legommender'
    user_encoder = 'user_encoder'
    item_encoder = 'item_encoder'
    predictor = 'predictor'


class Mediator:
    def __init__(self, legommender):
        from model.legommender import Legommender

        self._legommender = cast(Legommender, legommender)

        self._modules = {
            ModuleType.legommender: self._legommender,
            ModuleType.user_encoder: self._legommender.user_encoder,
            ModuleType.item_encoder: self._legommender.item_encoder,
            ModuleType.predictor: self._legommender.predictor,
        }

        for request_name in self._modules:
            if request_name is ModuleType.legommender:
                continue

            requester = self._modules[request_name]
            if not requester:
                continue

            requests = requester.request()  # type: Dict[str, list]
            if not requests:
                continue

            pnt(f'Requester {request_name} requests {requests}')

            for response_name in requests:
                responser = self._modules[response_name]
                response = responser.response(request_name, requests[response_name])
                requester.receive(response_name, response)
