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
import uuid
from subprocess import Popen, PIPE

from pigmento import pnt

from utils.config_init import AuthInit
from utils.server import Server


class Worker:
    def __init__(self):
        self.command = self.get_trainer_command()
        os.makedirs("log", exist_ok=True)

        self.log_file = os.path.join("log", f"log_{uuid.uuid4().hex}.log")
        self.pid_file = "worker.pid"

    @staticmethod
    def get_trainer_command():
        argv = sys.argv
        assert argv[0] == 'worker.py'
        argv = ' '.join(argv[1:])
        return f'python trainer.py {argv}'

    def run(self):
        """Run the trainer using nohup and save to a log file."""
        command = f"nohup {self.command} > {self.log_file} 2>&1 &"
        process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
        pid = process.pid

        # Save the PID for monitoring
        with open(self.pid_file, "w") as f:
            f.write(str(pid))
        pnt(f"Worker started with PID: {pid}. Log file: {self.log_file}")
        return pid

    def is_process_running(self):
        """Check if the process is still running using the saved PID."""
        try:
            with open(self.pid_file, "r") as f:
                pid = int(f.read().strip())
            # Check if the process is running
            os.kill(pid, 0)
            return True
        except (ValueError, ProcessLookupError):
            return False

    def monitor_and_upload(self):
        """Monitor the process until completion and upload results."""
        while self.is_process_running():
            pnt("Training in progress. Checking again in 1 minute...")
            time.sleep(60)  # Check every minute

        pnt("Training completed. Parsing log and uploading results.")
        base_dir, signature, log = self.parse_log()

        # Upload results after training is complete
        server = Server(
            auth=AuthInit.get('auth'),
            uri=AuthInit.get('uri'),
        )
        with open(self.log_file, "r") as f:
            server.post(
                signature=signature,
                command=self.command,
                log=log,
                base_dir=base_dir
            )
        pnt("Results uploaded successfully.")

    def parse_log(self):
        """Parse the log file to retrieve the base saving directory and signature."""
        base_dir = None
        signature = None

        with open(self.log_file, "r") as file:
            log = file.read()

        log = log.split('\n')
        log = [line for line in log if '\r' not in line]

        for line in log:
            if 'BASE DIR' in line:
                base_dir = line.split('BASE DIR: ')[1].strip()
            if 'SIGNATURE' in line:
                signature = line.split('SIGNATURE: ')[1].strip()

        return base_dir, signature, '\n'.join(log)


if __name__ == "__main__":
    worker = Worker()
    worker.run()
    worker.monitor_and_upload()
