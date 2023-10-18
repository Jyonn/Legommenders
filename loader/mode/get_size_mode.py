from pigmento import pnt

from loader.mode.base_mode import BaseMode


class GetSizeMode(BaseMode):
    load_model = False

    def work(self, *args, **kwargs):
        named_parameters = list(self.legommender.named_parameters())
        # filter out the parameters that don't require a gradient
        named_parameters = [(name, p) for (name, p) in named_parameters if p.requires_grad]
        # list of (name, parameter) pairs
        for (name, p) in named_parameters:
            pnt(name, p.data.shape)
        num_params = sum([p.numel() for (_, p) in named_parameters])
        # to a million
        num_params /= 1e6
        pnt(f'Number of parameters: {num_params:.2f}M')
