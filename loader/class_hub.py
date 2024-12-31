import glob
import importlib


class ClassHub:
    @staticmethod
    def operators():
        from model.operators.base_operator import BaseOperator
        return ClassHub(BaseOperator, 'model/operators', 'Operator')

    @staticmethod
    def predictors():
        from model.predictors.base_predictor import BasePredictor
        return ClassHub(BasePredictor, 'model/predictors', 'Predictor')

    @staticmethod
    def processors():
        from processor.base_processor import BaseProcessor
        return ClassHub(BaseProcessor, 'processor', 'Processor')

    @staticmethod
    def embedders():
        from embedder.base_embedder import BaseEmbedder
        return ClassHub(BaseEmbedder, 'embedder', 'Embedder')

    def __init__(self, base_class, module_dir: str, module_type: str):
        """
        @param base_class: e.g., BaseOperator, BasePredictor, BaseMode
        @param module_dir: e.g., model/operators, model/predictors, loader/modes
        @param module_type: e.g., Operator, Predictor, Mode
        """

        self.base_class = base_class
        self.module_dir = module_dir
        self.module_type = module_type.lower()
        self.upper_module_type = self.module_type.upper()[0] + self.module_type[1:]

        self.class_list = self.get_class_list()
        self.class_dict = dict()
        for class_ in self.class_list:
            name = class_.__name__
            name = name.replace(self.upper_module_type, '')
            self.class_dict[name] = class_

    def get_class_list(self):
        # file_paths = glob.glob('model/recommenders/*_model.py')
        file_paths = glob.glob(f'{self.module_dir}/*_{self.module_type}.py')
        class_list = []
        for file_path in file_paths:
            file_name = file_path.split('/')[-1].split('.')[0]
            module = importlib.import_module(f'{self.module_dir.replace("/", ".")}.{file_name}')

            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, self.base_class) and obj is not self.base_class:
                    class_list.append(obj)
        return class_list

    def __call__(self, name):
        return self.class_dict[name]

    def __getitem__(self, name):
        return self.class_dict[name]

    def __contains__(self, name):
        return name in self.class_dict
