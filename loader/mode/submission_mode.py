from pigmento import pnt

from loader.symbols import Symbols
from loader.mode.base_mode import BaseMode
from loader.mode.evaluate_mode import EvaluateMode
from utils.submission import Submission


class SubmissionMode(BaseMode):
    load_model = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluate = self.mode_hub(EvaluateMode)  # type: EvaluateMode

    def work(self):
        loader = self.manager.get_test_loader()
        self.legommender.eval()

        item_col, group_col = self.manager.cm.item_col, self.manager.cm.group_col
        score_series, col_series = self.evaluate(loader=loader, cols=[item_col, group_col])
        item_series, group_series = col_series[item_col], col_series[group_col]

        submission = Submission(
            ut=self.manager.test_ut,
            column_map=self.manager.cm,
        )

        export_dir = submission.run(
            scores=score_series,
            groups=group_series,
            items=item_series,
            model_name=self.manager.model.key,
        )

        pnt(f'export to {export_dir}')
