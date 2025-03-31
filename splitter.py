from base_lego import BaseLego
from model.operators.once_operator import OnceOperator
from utils.config_init import CommandInit


class Splitter(BaseLego):
    def run(self):
        item_op = self.legommender.item_op
        if not isinstance(item_op, OnceOperator):
            raise ValueError('item encoder is not a LMOperator')

        layers = map(int, self.config.layers.split('+'))
        layers = list(map(lambda x: x if x >= 0 else x + self.legommender.item_op.num_hidden_layers, layers))

        if not self.embed.embeddings:
            raise ValueError('please specify pretrained embedding configurations when using LM layer split')

        item_op.cache(layers)


if __name__ == '__main__':
    configuration = CommandInit(
        required_args=['data', 'model', 'embed', 'layers'],
        default_args=dict(
            exp='config/exp/default.yaml',
            # unused but required arguments
            hidden_size=256,
            batch_size=64,
        ),
    ).parse()

    splitter = Splitter(config=configuration)
    splitter.run()
