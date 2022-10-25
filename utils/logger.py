class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def __call__(self, s):
        with open(self.log_file, 'a+') as f:
            f.write(str(s) + '\n')
