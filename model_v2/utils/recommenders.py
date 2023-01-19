from model_v2.recommenders.bst_model import BSTModel
from model_v2.recommenders.dcn_model import DCNModel
from model_v2.recommenders.din_model import DINModel
from model_v2.recommenders.fancy_dcn_model import FancyDCNModel
from model_v2.recommenders.lstur_model import LSTURModel
from model_v2.recommenders.naml_model import NAMLModel
from model_v2.recommenders.nrms_model import NRMSModel
from model_v2.recommenders.plmnr_dcn_model import PLMNRDCNModel
from model_v2.recommenders.plmnr_din_model import PLMNRDINModel
from model_v2.recommenders.plmnr_fancy_dcn_model import PLMNRFancyDCNModel
from model_v2.recommenders.plmnr_lstur_model import PLMNRLSTURModel
from model_v2.recommenders.plmnr_naml_model import PLMNRNAMLModel
from model_v2.recommenders.plmnr_nrms_model import PLMNRNRMSModel
from model_v2.recommenders.prec_model import PRECModel

recommender_list = [
    NRMSModel,
    PRECModel,
    PLMNRDCNModel,
    PLMNRNRMSModel,
    PLMNRNAMLModel,
    PLMNRLSTURModel,
    PLMNRFancyDCNModel,
    PLMNRDINModel,
    FancyDCNModel,
    DCNModel,
    NAMLModel,
    LSTURModel,
    DINModel,
    BSTModel,
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
