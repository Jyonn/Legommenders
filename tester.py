from pigmento import pnt

from base_lego import BaseLego
from loader.env import Env
from utils import bars
from utils.config_init import CommandInit


class Tester(BaseLego):
    def test(self):
        loader = self.manager.get_test_loader()
        results = self.evaluate(loader, metrics=self.exp.metrics, bar=bars.TestBar())

        with open(Env.path_hub.result_path, 'w') as f:
            for metric in results:
                pnt(f'{metric}: {results[metric]:.4f}')
                f.write(f'{metric},{results[metric]:.4f}\n')

    def run(self):
        self.load()
        self.test()


if __name__ == '__main__':
    configuration = CommandInit(
        required_args=['data', 'model'],
        default_args=dict(
            exp='config/exp/default.yaml',
            embed='config/embed/null.yaml',
            hidden_size=256,
            item_hidden_size='${hidden_size}$',
        ),
    ).parse()

    tester = Tester(config=configuration)
    tester.run()
