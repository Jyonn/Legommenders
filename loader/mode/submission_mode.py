from pigmento import pnt

from loader.meta import Phases
from loader.mode.base_mode import BaseMode
from loader.mode.evaluate_mode import EvaluateMode
from utils.submission import Submission


class SubmissionMode(BaseMode):
    load_model = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluate = self.mode_hub(EvaluateMode)  # type: EvaluateMode

    def work(self):
        loader = self.controller.get_loader(Phases.test).test()
        self.legommender.eval()

        item_col, group_col = self.controller.column_map.candidate_col, self.controller.column_map.group_col
        score_series, col_series = self.evaluate(loader=loader, cols=[item_col, group_col])
        item_series, group_series = col_series[item_col], col_series[group_col]

        submission = Submission(
            depot=self.controller.depots[Phases.test],
            column_map=self.controller.column_map,
        )

        export_dir = submission.run(
            scores=score_series,
            groups=group_series,
            items=item_series,
            model_name=self.controller.model.name,
        )

        pnt(f'export to {export_dir}')
