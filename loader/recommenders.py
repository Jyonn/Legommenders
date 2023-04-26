from model.recommenders.bst_model import BSTModel
from model.recommenders.dcn_model import DCNModel
from model.recommenders.deepfm_model import DeepFMModel
from model.recommenders.din_model import DINModel
from model.recommenders.fancy_dcn_model import FancyDCNModel
from model.recommenders.lstur_model import LSTURModel
from model.recommenders.naml_dcn_model import NAMLDCNModel
from model.recommenders.naml_fancy_dcn_model import NAMLFancyDCNModel
from model.recommenders.naml_model import NAMLModel
from model.recommenders.nrms_model import NRMSModel
from model.recommenders.plmnr_bst_model import PLMNRBSTModel
from model.recommenders.plmnr_dcn_model import PLMNRDCNModel
from model.recommenders.plmnr_din_model import PLMNRDINModel
from model.recommenders.plmnr_fancy_dcn_model import PLMNRFancyDCNModel
from model.recommenders.plmnr_lstur_model import PLMNRLSTURModel
from model.recommenders.plmnr_naml_model import PLMNRNAMLModel
from model.recommenders.plmnr_nrms_model import PLMNRNRMSModel
from model.recommenders.pnn_model import PNNModel

recommender_list = [
    NRMSModel,
    PLMNRDCNModel,
    PLMNRBSTModel,
    PLMNRNRMSModel,
    PLMNRNAMLModel,
    PLMNRLSTURModel,
    PLMNRDINModel,
    DCNModel,
    DeepFMModel,
    PNNModel,
    NAMLModel,
    LSTURModel,
    DINModel,
    BSTModel,

    NAMLDCNModel,
    FancyDCNModel,
    NAMLFancyDCNModel,
    PLMNRFancyDCNModel,
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
