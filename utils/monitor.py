from loader.symbols import Symbols


class Monitor:
    def __init__(self, minimize, patience=2):
        self.patience = patience
        self.best_value = None
        self.minimize = minimize
        self.best_index = 0
        self.current_index = -1

    def push(self, value):
        self.current_index += 1

        if self.best_value is None:
            self.best_value = value
            return Symbols.best

        if self.minimize ^ (value > self.best_value):
            self.best_value = value
            self.best_index = self.current_index
            return Symbols.best

        if self.current_index - self.best_index >= self.patience:
            return Symbols.stop
        return Symbols.skip
