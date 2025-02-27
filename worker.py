import concurrent.futures
import random
import sys
import time
from subprocess import Popen

from oba import Obj
from pigmento import pnt
from unitok import JsonHandler

import trainer
from utils import function
from utils.config_init import CommandInit
from utils.gpu import GPU
from utils.server import Server, ExperimentBody, EvaluationBody


class Worker:
    SEED_START = 2023

    def __init__(self, config):
        self.server = Server.auto_auth()

        self.jobs = self.load_jobs(config.jobs)
        self.replicate = config.replicate
        self.seeds = self.get_seeds(config.seeds, self.replicate)
        self.num_workers = config.num_workers or len(GPU.get_gpus())

        self.all_evaluations = self.get_all_evaluations()

    @staticmethod
    def load_jobs(job_file):
        with open(job_file) as f:
            return f.read().strip().split('\n')

    def get_all_evaluations(self):
        evaluations = self.server.get_all_evaluations()
        evaluation_dict = dict()
        for evaluation in evaluations.body:
            evaluation = EvaluationBody(evaluation)
            evaluation_dict[evaluation.command] = []
            for experiment in evaluation.experiments:  # type: ExperimentBody
                if experiment.is_completed:
                    evaluation_dict[evaluation.command].append(experiment.seed)
        return evaluation_dict

    @staticmethod
    def get_seeds(seeds, replicate):
        if seeds is not None:
            if len(seeds) != replicate:
                raise ValueError('seeds should have a length of replicate')
            seeds = [int(seed) for seed in seeds.split('+')]
            return seeds
        return list(range(Worker.SEED_START, Worker.SEED_START + replicate))

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

    def run_experiment(self, command, signature, seed):
        experiment = self.server.create_or_get_experiment(signature, seed)
        if not experiment.ok:
            raise ValueError(f'failed to create experiment: {experiment.msg}')
        session = experiment.body

        experiment = self.server.get_experiment_info(session)
        experiment = ExperimentBody(experiment.body)
        if experiment.is_completed:
            pnt(f"Experiment ({signature}, {seed}) is already completed.")
            return

        command = f'{command} --seed {seed} --session {session}'

        required_memory = self.get_memory_required(command)
        while True:
            time.sleep(random.randint(60, 300))
            free_memory = GPU.get_maximal_free_gpu()
            if free_memory >= required_memory:
                break

        pnt(f'running command: {command}')
        process = Popen(command, shell=True)
        process.wait()

    @staticmethod
    def get_memory_required(command: str):
        if 'llama' in command.lower():
            return 70_000
        if 'bert' in command.lower():
            return 24_000
        return 10_000

    def run_evaluation(self, job: str):
        time.sleep(random.randint(0, 10))

        job = job.strip()
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
        configuration = JsonHandler.dumps(Obj.raw(configuration))

        evaluation = self.server.create_or_get_evaluation(
            signature=signature,
            command=command,
            configuration=configuration,
        )
        if not evaluation.ok:
            raise ValueError(f'failed to create evaluation: {evaluation.msg}')

        for seed in self.seeds:
            if command in self.all_evaluations:
                evaluation = self.all_evaluations[command]
                if seed in evaluation:
                    pnt(f"Experiment ({signature}, {seed}) is already done.")
                    continue

            self.run_experiment(command, signature, seed)

    def run(self):
        # for job in self.jobs:
        #     self.run_evaluation(job)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all jobs to the executor.
            future_to_job = {executor.submit(self.run_evaluation, job): job for job in self.jobs}

            # Process results as jobs complete.
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    future.result()  # This will re-raise any exception caught in run_job.
                except Exception as e:
                    pnt(f"Job {job} generated an exception: {e}")


if __name__ == "__main__":
    configuration = CommandInit(
        required_args=['jobs'],
        default_args=dict(
            replicate=5,
            seeds=None,
            num_workers=None,
        ),
    ).parse()

    worker = Worker(config=configuration)
    worker.run()
