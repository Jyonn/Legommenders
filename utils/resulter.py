import os


class Resulter:
    def __init__(self, dataset, model, metrics, config, parameters: list[str]):
        self.dataset = dataset
        self.model = model
        self.metrics = metrics

        # create result csv
        self.result_csv = f'resulter_{self.dataset}.csv'
        short_parameters = []
        for parameter in parameters:
            pos = parameter.rfind('.')
            short_parameters.append(parameter[pos + 1:])
