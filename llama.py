from transformers import OpenLlamaModel


class LLaMA:
    def __init__(self):
        self.model = OpenLlamaModel.from_pretrained('/data2/liming/')
