class Status:
    def __init__(self):
        self.is_training = True
        self.is_evaluating = False
        self.is_testing = False

    def train(self):
        self.is_training = True
        self.is_evaluating = False
        self.is_testing = False

    def eval(self):
        self.is_training = False
        self.is_evaluating = True
        self.is_testing = False

    def test(self):
        self.is_training = False
        self.is_evaluating = False
        self.is_testing = True
