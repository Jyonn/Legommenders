import os

import torch.cuda
from pigmento import pnt


class GPU:
    @classmethod
    def parse_gpu_info(cls, line, args):
        def to_number(v):
            return float(v.upper().strip().replace('MIB', '').replace('W', ''))

        def processor(k, v):
            return (int(to_number(v)) if 'Not Support' not in v else 1) if k in params else v.strip()

        params = ['memory.free', 'memory.total', 'power.draw', 'power.limit']
        return {k: processor(k, v) for k, v in zip(args, line.strip().split(','))}

    @classmethod
    def get_gpus(cls):
        args = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(args))
        results = os.popen(cmd).readlines()
        return [cls.parse_gpu_info(line, args) for line in results]

    @classmethod
    def auto_choose(cls, torch_format=False):
        if not torch.cuda.is_available():
            pnt('not support cuda')
            if torch_format:
                pnt('switch to CPU')
                return "cpu"
            return -1

        gpus = cls.get_gpus()
        chosen_gpu = sorted(gpus, key=lambda d: d['memory.free'], reverse=True)[0]
        pnt('choose', chosen_gpu['index'], 'GPU with',
            chosen_gpu['memory.free'], '/', chosen_gpu['memory.total'], 'MB')
        if torch_format:
            return "cuda:" + str(chosen_gpu['index'])
        return int(chosen_gpu['index'])


if __name__ == '__main__':
    GPU.auto_choose()
