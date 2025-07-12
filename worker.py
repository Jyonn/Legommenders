"""
worker.py

Batch-executor that orchestrates **multiple trainer runs** on a single
machine (typically equipped with several GPUs) *and* tracks their
progress via a remote lego-server instance.

High-level responsibilities
---------------------------
1. Read a plain-text *job list* (each line is a CLI snippet for
   `trainer.py`).
2. For every job:
   • create/lookup an *evaluation* entry on the server  
   • spin up `replicate` repetitions with different seeds
   • make sure the same (command, seed) combination is **not** executed
     twice (checks previously completed experiments stored on the
     server).
3. Use a `ThreadPoolExecutor` so that multiple jobs can be processed
   concurrently (GPU assignment is handled by the internal heuristics
   that poll available memory).
4. Add a small waiting loop that delays execution until a GPU has at
   least `<required_memory>` MiB of free memory.

Notes
-----
• The script does **not** directly import heavy ML libraries to keep
  start-up time low. It only imports `trainer` to access helper
  functions (`get_configurations`) and to provide a proper
  `get_signature` implementation.

• Memory heuristics are admittedly crude: they just look for certain
  substrings (`llama`, `bert`, …) inside the command and return a hard
  coded threshold.  Feel free to adapt to your own hardware.

• All interaction with the remote service is performed via
  `utils.server.Server`.
"""

from __future__ import annotations

import concurrent.futures
import random
import time
from subprocess import Popen
from typing import List, Dict

from pigmento import pnt

# Project utilities
import trainer                                   # only lightweight imports
from utils import function, io
from utils.config_init import CommandInit
from utils.gpu import GPU
from utils.server import Server, ExperimentBody, EvaluationBody


class Worker:
    """
    Orchestrates *many* independent `trainer.py` runs.
    """

    SEED_START = 2023  # default start value if --seeds is not provided

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(self, config):
        self.server = Server.auto_auth()

        self.jobs: List[str] = self.load_jobs(config.jobs)
        self.replicate: int = config.replicate
        self.seeds: List[int] = self.get_seeds(config.seeds, self.replicate)
        self.num_workers: int = config.num_workers or len(GPU.get_gpus())

        self.all_evaluations: Dict[str, List[int]] = self.get_all_evaluations()

    # ------------------------------------------------------------------ #
    # Helper: Job list                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_jobs(job_file: str) -> List[str]:
        """
        Each line in `job_file` is expected to be a valid argument string
        for `trainer.py`.

        Example line
        ------------
        --data config/data/mind.yaml --model config/model/naml.yaml
        """
        return io.file_load(job_file).strip().split('\n')

    # ------------------------------------------------------------------ #
    # Helper: Server bookkeeping                                         #
    # ------------------------------------------------------------------ #
    def get_all_evaluations(self) -> Dict[str, List[int]]:
        """
        Build a local cache that maps

            command_string -> [list of completed seeds]

        so we can skip duplicates early.
        """
        evaluations = self.server.get_all_evaluations()
        evaluation_dict: Dict[str, List[int]] = {}
        for evaluation in evaluations:                          # type: EvaluationBody
            evaluation_dict[evaluation.command] = []
            for experiment in evaluation.experiments:           # type: ExperimentBody
                if experiment.is_completed:
                    evaluation_dict[evaluation.command].append(experiment.seed)
        return evaluation_dict

    # ------------------------------------------------------------------ #
    # Helper: Seed generation                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_seeds(seeds: str | None, replicate: int) -> List[int]:
        """
        Return a list of distinct seeds that should be used for the
        `replicate` runs.

        • If `--seeds` was supplied, we parse the `a+b+c` formatted
          string and perform sanity checks.

        • Otherwise we return `[SEED_START, SEED_START+1, …]`.
        """
        if seeds is not None:
            seed_list = [int(seed) for seed in seeds.split('+')]
            if len(seed_list) != replicate:
                raise ValueError('`--seeds` length must match `--replicate`')
            return seed_list
        return list(range(Worker.SEED_START, Worker.SEED_START + replicate))

    # ------------------------------------------------------------------ #
    # Experiment execution                                               #
    # ------------------------------------------------------------------ #
    def run_experiment(self, command: str, signature: str, seed: int) -> None:
        """
        Safely launch a **single** (command, seed) run.

        1. Register the experiment on the server (or fetch existing)
        2. Wait until enough GPU memory is available
        3. Spawn a child process via `subprocess.Popen`
        """
        experiment_reply = self.server.create_or_get_experiment(signature, seed)
        if not experiment_reply.ok:
            raise ValueError(f'Failed to create experiment: {experiment_reply.msg}')
        session = experiment_reply.body

        experiment_info = self.server.get_experiment_info(session)
        experiment = ExperimentBody(experiment_info.body)
        if experiment.is_completed:
            pnt(f"Experiment ({signature}, {seed}) is already completed.")
            return

        # Append seed + session to the CLI command
        command = f'{command} --seed {seed} --session {session}'

        # -------- Wait for sufficient free GPU memory ------------------ #
        required_memory = self.get_memory_required(command)
        while True:
            time.sleep(random.randint(60, 300))      # poll every 1-5 minutes
            free_memory = GPU.get_maximal_free_gpu()  # MiB
            if free_memory >= required_memory:
                break

        pnt(f'Running command: {command}')
        process = Popen(command, shell=True)
        process.wait()                               # blocking – okay inside thread

    # crude heuristics; adjust to your own GPUs / models
    @staticmethod
    def get_memory_required(command: str) -> int:
        cmd_lower = command.lower()
        if 'llama' in cmd_lower:
            return 70_000  # 70 GB
        if 'bert' in cmd_lower:
            return 24_000
        if 'miner' in cmd_lower:
            return 24_000
        return 15_000

    # ------------------------------------------------------------------ #
    # Evaluation entry point                                             #
    # ------------------------------------------------------------------ #
    def run_evaluation(self, job: str) -> None:
        """
        Perform all `replicate` runs for a *single* line from the job
        file.
        """
        # Stagger start-up a bit so not all threads hit the server at once
        time.sleep(random.randint(0, 10))

        job = job.strip()
        # ------------------------------------------------------------------ #
        # 1. Compute signature (same helper as trainer.py)                    #
        # ------------------------------------------------------------------ #
        job_split = job.split()
        kwargs = function.argparse(job_split)
        configuration = trainer.get_configurations(kwargs)
        signature = function.get_signature(
            data=configuration.data,
            embed=configuration.embed,
            model=configuration.model,
            exp=configuration.exp,
        )
        command = f'python trainer.py {job}'
        configuration_json = io.json_dumps(configuration())

        # ------------------------------------------------------------------ #
        # 2. Register *evaluation* (a group of experiments)                   #
        # ------------------------------------------------------------------ #
        evaluation_reply = self.server.create_or_get_evaluation(
            signature=signature,
            command=command,
            configuration=configuration_json,
        )
        if not evaluation_reply.ok:
            raise ValueError(f'Failed to create evaluation: {evaluation_reply.msg}')

        # ------------------------------------------------------------------ #
        # 3. Launch replicate experiments                                     #
        # ------------------------------------------------------------------ #
        for seed in self.seeds:
            if command in self.all_evaluations:
                if seed in self.all_evaluations[command]:
                    pnt(f"Experiment ({signature}, {seed}) is already done.")
                    continue  # skip duplicate

            self.run_experiment(command, signature, seed)

    # ------------------------------------------------------------------ #
    # Main loop                                                          #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """
        Process all jobs using a thread pool (= concurrency on *CPU*
        bound parts).  Each thread spawns its own trainer *sub-process*,
        so GPU utilisation is handled at a lower level.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Map Future -> job string
            future_to_job = {
                executor.submit(self.run_evaluation, job): job
                for job in self.jobs
            }

            # Collect results / propagate exceptions
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    future.result()   # will re-raise exception from thread
                except Exception as e:
                    pnt(f"Job {job} generated an exception: {e}")


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    configuration = CommandInit(
        required_args=['jobs'],     # path to job-file
        default_args=dict(
            replicate=5,            # how many seeds per job
            seeds=None,             # explicit seed list (e.g. "1+2+3+4+5")
            num_workers=None,       # defaults to #available GPUs
        ),
    ).parse()

    worker = Worker(config=configuration)
    worker.run()
