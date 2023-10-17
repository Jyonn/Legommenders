import time

from pigmento import pnt


class StatusTimer:
    def __init__(self):
        self.total_time = 0
        self.start_time = None
        self.timing = False
        self.count = 0

    def run(self):
        crt_time = time.time()
        if self.timing:
            self.total_time += crt_time - self.start_time
            self.timing = False
            self.count += 1
        else:
            self.timing = True
            self.start_time = crt_time

    def clear(self):
        self.total_time = 0
        self.start_time = None
        self.timing = False
        self.count = 0

    def avgms(self):
        if not self.count:
            return 'unavailable'
        return self.total_time * 1000 / self.count


class Timer:
    def __init__(self, activate=True):
        self.status_dict = dict()  # type: dict[str, StatusTimer]
        self.activate = activate

    def run(self, status: str):
        if not self.activate:
            return
        if status not in self.status_dict:
            self.status_dict[status] = StatusTimer()
        self.status_dict[status].run()

    def __call__(self, status: str):
        return self.run(status)

    def clear(self):
        for status in self.status_dict:
            self.status_dict[status].clear()

    def summarize(self):
        for status in self.status_dict:
            pnt(f'status: {status}, '
                f'avg time: {self.status_dict[status].avgms():.4f}ms, '
                f'total time: {self.status_dict[status].total_time:.4f}s')
