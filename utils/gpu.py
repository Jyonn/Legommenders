import os
import torch
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
    def get_maximal_free_gpu(cls):
        gpus = cls.get_gpus()
        gpu = sorted(gpus, key=lambda d: d['memory.free'], reverse=True)[0]
        return int(gpu['memory.free'])

    @classmethod
    def auto_choose(cls, torch_format=False):
        # 如果支持 CUDA，则使用 nvidia-smi 获取信息
        if torch.cuda.is_available():
            gpus = cls.get_gpus()
            chosen_gpu = sorted(gpus, key=lambda d: d['memory.free'], reverse=True)[0]
            pnt('choose', chosen_gpu['index'], 'GPU with',
                chosen_gpu['memory.free'], '/', chosen_gpu['memory.total'], 'MB')
            if torch_format:
                return "cuda:" + str(chosen_gpu['index'])
            return int(chosen_gpu['index'])
        # 否则检查是否支持 MPS（针对 mac M 系列 GPU）
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pnt('MPS available: using mac M series GPU')
            if torch_format:
                return "mps"
            return -1  # 如果不需要 torch 格式，可自行修改返回值
        else:
            pnt('not support cuda or mps, switch to CPU')
            if torch_format:
                return "cpu"
            return -1


if __name__ == '__main__':
    device = GPU.auto_choose(torch_format=True)
    pnt("Selected device:", device)
