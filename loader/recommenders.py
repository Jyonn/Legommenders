import glob
import importlib

from model.recommenders.base_recommender import BaseRecommender


class Recommenders:
    def __init__(self):
        self.recommender_list = self.get_recommender_list()
        self.recommender_dict = dict()
        for recommender in self.recommender_list:
            name = recommender.__name__
            name = name.replace('Model', '')
            self.recommender_dict[name] = recommender

    @staticmethod
    def get_recommender_list():
        file_paths = glob.glob('model/recommenders/*_model.py')
        recommender_list = []
        for file_path in file_paths:
            file_name = file_path.split('/')[-1].split('.')[0]
            module = importlib.import_module(f'model.recommenders.{file_name}')

            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and issubclass(obj, BaseRecommender) and obj is not BaseRecommender:
                    recommender_list.append(obj)
        return recommender_list

    def __call__(self, name):
        return self.recommender_dict[name]
