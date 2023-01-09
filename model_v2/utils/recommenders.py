from model_v2.recommenders.nrms_model import NRMSModel

recommender_list = [
    NRMSModel,
]


class Recommenders:
    def __init__(self):
        self.recommender_list = recommender_list
        self.recommender_dict = dict()
        for recommender in self.recommender_list:
            name = recommender.__name__
            name = name.replace('Model', '')
            self.recommender_dict[name] = recommender

    def get(self, name):
        return self.recommender_dict[name]

    def __call__(self, name):
        return self.get(name)
