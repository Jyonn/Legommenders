from pigmento import pnt

from base_lego import BaseLego
from utils.config_init import CommandInit


class Sizer(BaseLego):
    def run(self):
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


if __name__ == '__main__':
    configuration = CommandInit(
        required_args=['data', 'model'],
        default_args=dict(
            exp='config/exp/default.yaml',
            embed='config/embed/null.yaml',
            hidden_size=256,
            item_hidden_size='${hidden_size}$',
            item_page_size=64,
            batch_size=64,
        ),
    ).parse()

    sizer = Sizer(config=configuration)
    sizer.run()
