"""
server.py

A *very thin* HTTP client wrapper that talks to a custom backend
(here named “Lego” in the configuration).  The module provides:

    • Small data-classes that convert raw JSON dictionaries into Python
      objects (BaseResp, ExperimentBody, EvaluationBody).
    • Utility for automatic authentication via `utils.config_init.AuthInit`.
    • `Server` class with helper methods covering the REST API surface:
         – CRUD for “evaluations” and “experiments”
         – Pagination handling for listing evaluations
         – Convenience logging of payload size & pretty-printing with
           `pigmento.pnt`.

None of the methods perform any retry / robustness logic—this is kept
intentionally minimal and should be extended if used in production.
"""

import os
from typing import Dict, Any, Iterator, Optional

import requests
from pigmento import pnt

from utils.config_init import AuthInit


# =============================================================================
#                         Lightweight response wrappers
# =============================================================================
class BaseResp:
    """
    A convenience wrapper around the plain JSON dict returned by the server.
    Only copies a handful of well-known keys to attributes and provides an
    `ok` property for quick success checks.
    """

    def __init__(self, resp: Dict[str, Any]):
        self.msg: Optional[str] = resp.get("msg")
        self.identifier: Optional[str] = resp.get("identifier")
        self.append_msg: Optional[str] = resp.get("append_msg")
        self.debug_msg: Optional[str] = resp.get("debug_msg")
        self.code: Optional[int] = resp.get("code")
        self.body: Any = resp.get("body")
        self.http_code: Optional[int] = resp.get("http_code")

    # ---------------------------------------------------------------------
    # Helper: success flag
    # ---------------------------------------------------------------------
    @property
    def ok(self) -> bool:
        """
        According to the backend specification a successful response has
        an identifier equal to the literal string "OK".
        """
        return self.identifier == "OK"


class ExperimentBody:
    """
    Client-side representation of an “Experiment” object as returned by the
    server.  All attributes are optional because the server may evolve.
    """

    def __init__(self, body: Dict[str, Any]):
        self.signature = body.get("signature")
        self.seed = body.get("seed")
        self.session = body.get("session")
        self.log = body.get("log")
        self.performance = body.get("performance")
        self.is_completed = body.get("is_completed")
        self.created_at = body.get("created_at")
        self.pid = body.get("pid")


class EvaluationBody:
    """
    Client-side representation of an “Evaluation” + its nested experiments.
    """

    def __init__(self, body: Dict[str, Any]):
        self.signature = body.get("signature")
        self.command = body.get("command")
        self.configuration = body.get("configuration")
        self.created_at = body.get("created_at")
        self.modified_at = body.get("modified_at")
        self.comment = body.get("comment")

        # Convert nested experiment dictionaries into ExperimentBody objects
        self.experiments = [ExperimentBody(e) for e in body.get("experiments", [])]


# =============================================================================
#                                 Server
# =============================================================================
class Server:
    """
    Very small wrapper around `requests` that handles authentication headers,
    logging and convenience methods for the specific API endpoints used by
    this project.

    Parameters
    ----------
    uri  : str
        Base URI of the Lego server, e.g. "https://api.lego.ai".
    auth : str
        Authentication token that the server expects in the "Authentication"
        header.
    """

    def __init__(self, uri: str, auth: str):
        self.uri = uri
        self.auth = auth
        self.pid = os.getpid()  # process identifier of the current Python worker

    # ---------------------------------------------------------------------
    # Automatic authentication from config file
    # ---------------------------------------------------------------------
    @classmethod
    def auto_auth(cls) -> "Server":
        """
        Construct a `Server` instance by pulling `lego_uri` and `lego_auth`
        values from the project's configuration system.
        """
        uri = AuthInit.get("lego_uri")
        auth = AuthInit.get("lego_auth")
        return cls(uri=uri, auth=auth)

    # ---------------------------------------------------------------------
    # Helper: compute rough payload size (bytes)
    # ---------------------------------------------------------------------
    @staticmethod
    def calculate_bytes(data: Dict[str, Any]) -> int:
        """
        Naively approximates the byte size of the JSON payload by summing the
        string length of keys and values.  Good enough for logging purposes.
        """
        return sum(len(str(key)) + len(str(value)) for key, value in data.items())

    # ---------------------------------------------------------------------
    # Low-level HTTP wrappers
    # ---------------------------------------------------------------------
    def post(self, uri: str, data: Dict[str, Any]) -> BaseResp:
        total_bytes = self.calculate_bytes(data)
        pnt(f"Uploading {total_bytes} bytes of data to {uri}")

        with requests.post(
            uri,
            headers={"Authentication": self.auth},
            json=data,
        ) as response:
            return BaseResp(response.json())

    def put(self, uri: str, data: Dict[str, Any]) -> BaseResp:
        total_bytes = self.calculate_bytes(data)
        pnt(f"Uploading {total_bytes} bytes of data to {uri}")

        with requests.put(
            uri,
            headers={"Authentication": self.auth},
            json=data,
        ) as response:
            return BaseResp(response.json())

    def delete(self, uri: str) -> BaseResp:
        pnt(f"Sending delete request to {uri}")
        with requests.delete(
            uri,
            headers={"Authentication": self.auth},
        ) as response:
            return BaseResp(response.json())

    def get(self, uri: str, query: Dict[str, Any]) -> BaseResp:
        pnt(f"Sending query request to {uri} with {query}")
        with requests.get(
            uri,
            headers={"Authentication": self.auth},
            params=query,
        ) as response:
            return BaseResp(response.json())

    # ---------------------------------------------------------------------
    # High-level domain-specific helpers
    # ---------------------------------------------------------------------
    def get_all_evaluations(self) -> Iterator[EvaluationBody]:
        """
        Yields *all* evaluations using the server's pagination.  The server is
        expected to return a JSON with keys 'total_page' and 'evaluations'.
        """
        total_page = None
        current_page = 1

        while total_page is None or current_page <= total_page:
            query = {"page": current_page}
            response = self.get(f"{self.uri}/evaluations/", query)

            if response.ok:
                total_page = response.body["total_page"]
                for evaluation in response.body["evaluations"]:
                    yield EvaluationBody(evaluation)
                current_page += 1
            else:
                raise ValueError("Unable to fetch evaluations: " + (response.msg or ""))

    def get_experiment_info(self, session: str) -> BaseResp:
        query = {"session": session}
        return self.get(f"{self.uri}/experiments/", query)

    # --------------- CRUD for Evaluations ----------------------------------
    def create_or_get_evaluation(
        self, signature: str, command: str, configuration: str
    ) -> BaseResp:
        data = {
            "signature": signature,
            "command": command,
            "configuration": configuration,
        }
        return self.post(f"{self.uri}/evaluations/", data)

    def delete_evaluation(self, signature: str) -> BaseResp:
        return self.delete(f"{self.uri}/evaluations/{signature}")

    # --------------- CRUD for Experiments ----------------------------------
    def create_or_get_experiment(self, signature: str, seed: int) -> BaseResp:
        data = {"signature": signature, "seed": seed}
        return self.post(f"{self.uri}/experiments/", data)

    def register_experiment(self, session: str) -> BaseResp:
        data = {"pid": self.pid}
        return self.post(f"{self.uri}/experiments/{session}/register", data)

    def complete_experiment(
        self, session: str, log: str, performance: str
    ) -> BaseResp:
        data = {
            "session": session,
            "log": log,
            "performance": performance,
        }
        return self.put(f"{self.uri}/experiments/", data)


# =============================================================================
#                                 Demo
# =============================================================================
if __name__ == "__main__":
    # Fetch auth credentials from the config file
    server = Server(
        uri=AuthInit.get("uri"),
        auth=AuthInit.get("auth"),
    )

    # Example usage: create or fetch an evaluation
    evaluation_resp = server.create_or_get_evaluation(
        signature="123456",
        command="python train.py --model abc",
        configuration='{"learning_rate": 1e-3}',
    )

    if evaluation_resp.ok:
        pnt("Evaluation up-to-date:", evaluation_resp.body)
    else:
        pnt("Something went wrong:", evaluation_resp.msg)
