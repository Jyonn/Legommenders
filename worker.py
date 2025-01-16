"""
Worker is an automated process that trains a model, targets the logs and evaluates the model.

Worker will call trainer.py to train the model, which will also generate logs and evaluation results.
Based on the stdout of the trainer, worker can retrieve the base saving directory, and the training signature.
Worker will upload these meta information to the database, including:
    - model name and config
    - data name and config
    - embed name and config
    - exp config
    - training signature
    - log file
    - evaluation results

In the stdout of the trainer, first several lines will indicate the training signature, and the base saving directory:


    [W] |Instance| It is recommended to declare tokenizers and vocabularies in a UniTok context, using `with UniTok() as ut:`
    [00:00:00] |Trainer| START TIME: 2025-01-08 00:14:18.586587
    [00:00:00] |Trainer| SIGNATURE: EDWmZVNS
    [00:00:00] |Trainer| BASE DIR: checkpoints/mind/NRMS

"""
import sys
import time
import os
from subprocess import Popen
from pigmento import pnt

from trainer import get_configurations
from utils import function
from utils.config_init import AuthInit
from utils.path_hub import PathHub
from utils.server import Server


class Worker:
    def __init__(self, config):
        self.config = config
        self.data, self.model, self.embed, self.exp = \
            config.data, config.model, config.embed, config.exp
        self.signature = function.get_signature(
            data=self.data,
            model=self.model,
            embed=self.embed,
            exp=self.exp
        )

        self.path_hub = PathHub(
            data_name=self.data.name,
            model_name=self.model.name,
            signature=self.signature
        )

        self.metric = self.exp.store.metric

        self.command_args = self.get_trainer_command_args()

        os.makedirs("log", exist_ok=True)
        self.log_file = os.path.join("log", f"{self.signature}.log")

    @staticmethod
    def get_trainer_command_args():
        """
        Convert user arguments into a list:
        sys.argv[0] is worker.py, so skip it,
        then combine the rest into a trainer command.
        """
        argv = sys.argv
        assert argv[0] == 'worker.py'
        # e.g., if user runs:
        #   python worker.py --data config/data/mind.yaml --model config/model/naml.yaml
        # then argv is ["worker.py", "--data", "config/data/mind.yaml", "--model", "config/model/naml.yaml"]
        trainer_argv = argv[1:]
        # We build a list: ["python", "trainer.py", "--data", "config/data/mind.yaml", "--model", ...]
        return ["python", "trainer.py"] + trainer_argv

    def run(self):
        """Run the trainer in the background with Python as the parent."""
        with open(self.log_file, "wb") as lf:
            # Use a list of args instead of shell=True to avoid an extra shell.
            process = Popen(
                self.command_args,
                stdout=lf,     # send stdout to log file
                stderr=lf,     # send stderr to log file
                preexec_fn=os.setpgrp  # detach from controlling terminal if needed
            )
        pid = process.pid

        pnt(f"Worker started with PID: {pid}. Writing logs to: {self.log_file}")
        return process

    def monitor(self, process):
        """Monitor the process until completion."""
        while True:
            # poll() returns None if process is still running
            if process.poll() is None:
                pnt("Training in progress. Checking again in 1 minute...")
                time.sleep(60)
            else:
                pnt(f"Training process (PID {process.pid}) has finished.")
                break

    def read(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()
        return {'error': 'file not found'}

    def upload(self):
        """Parse log for BASE DIR and SIGNATURE, then upload results."""
        pnt("Parsing log and uploading results.")
        log_content = self.parse_log()

        server = Server(
            auth=AuthInit.get('auth'),
            uri=AuthInit.get('uri'),
        )
        server.post(
            signature=self.signature,
            command=" ".join(self.command_args),  # store the command
            log=log_content,
            config=self.read(self.path_hub.cfg_path),
            performance=self.read(self.path_hub.result_path),
        )
        pnt("Results uploaded successfully.")

    def parse_log(self):
        """Parse the log file to retrieve the base saving directory and signature."""
        # Read the entire log file as binary
        with open(self.log_file, 'rb') as f:
            log_bin = f.read()

        # Clean up progress lines: if there's a \r in a line, keep only what's after the last \r
        lines = log_bin.split(b'\n')
        cleaned_lines = []
        for line in lines:
            if b'\r' in line:
                # Keep content after the last carriage return
                line = line[line.rfind(b'\r') + 1:]
            cleaned_lines.append(line)

        log_bin = b'\n'.join(cleaned_lines)
        log_str = log_bin.decode('utf-8', errors='replace')

        return log_str


if __name__ == "__main__":
    worker = Worker(config=get_configurations())
    worker.upload()
