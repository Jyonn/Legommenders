from model.inputer.natural_concat_inputer import NaturalConcatInputer


class LlamaConcatInputer(NaturalConcatInputer):
    @staticmethod
    def get_start_prompt():
        return [10130, 4274, 29901]

    @staticmethod
    def get_col_prompts():
        return dict(
            title=[529, 3257, 29958],
            abs=[529, 16595, 29958],
            cat=[529, 7320, 29958],
            subCat=[529, 1491, 7320, 29958],
        )


class BertConcatInputer(NaturalConcatInputer):
    @staticmethod
    def get_start_prompt():
        return [2739, 3720, 1024]

    @staticmethod
    def get_col_prompts():
        return dict(
            title=[1026, 2516, 1028],
            abs=[1026, 10061, 1028],
            cat=[1026, 4696, 1028],
            subCat=[1026, 4942, 16280, 20255, 2100, 1028],
        )
