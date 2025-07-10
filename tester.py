from pigmento import pnt

from base_lego import BaseLego
from loader.env import Env
from loader.symbols import Symbols
from utils import bars, io
from utils.config_init import CommandInit
from utils.timer import StatusTimer


class Tester(BaseLego):
    def test(self):
        loader = self.manager.get_test_loader()
        results = self.evaluate(loader, metrics=self.exp.metrics, bar=bars.TestBar())

        lines = []
        # with open(Env.path_hub.result_path, 'w') as f:
        for metric in results:
            pnt(f'{metric}: {results[metric]:.4f}')
            # f.write(f'{metric},{results[metric]:.4f}\n')
            lines.append(f'{metric}: {results[metric]:.4f}')

        io.file_save(Env.ph.result_path, '\n'.join(lines))
        return results

    def latency(self):
        Env.latency_timer.activate()
        Env.latency_timer.clear()

        try:
            self.test()
        except KeyboardInterrupt:
            st = Env.latency_timer.status_dict[Symbols.test]  # type: StatusTimer
            pnt(f'Total {st.count} steps, avg ms {st.avgms():.4f}')

    def run(self):
        self.load()

        if self.config.latency:
            self.latency()
        else:
            self.test()


if __name__ == '__main__':
    configuration = CommandInit(
        required_args=['data', 'model'],
        default_args=dict(
            exp='config/exp/default.yaml',
            embed='config/embed/null.yaml',
            hidden_size=256,
            item_hidden_size='${hidden_size}$',
            latency=False,
        ),
    ).parse()

    tester = Tester(config=configuration)
    tester.run()
