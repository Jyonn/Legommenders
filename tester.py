from pigmento import pnt

from base_lego import BaseLego
from utils.config_init import CommandInit


class Tester(BaseLego):
    def test(self):
        loader = self.manager.get_test_loader()

        results = self.evaluate(loader, metrics=self.exp.metrics)
        for metric in results:
            pnt(f'{metric}: {results[metric]:.4f}')

    def run(self):
        self.load()
        self.test()


if __name__ == '__main__':
    configuration = CommandInit(
        required_args=['data', 'model', 'exp'],
        default_args=dict(
            embed='config/embed/null.yaml',
            hidden_size=256,
            item_hidden_size='${hidden_size}$',
        ),
    ).parse()

    tester = Tester(config=configuration)
    tester.run()
