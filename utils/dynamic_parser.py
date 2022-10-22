import sys


class DynamicParser:
    @classmethod
    def parse(cls):
        arguments = sys.argv[1:]
        kwargs = {}

        key = None
        for arg in arguments:
            if key is not None:
                kwargs[key] = arg
                key = None
            else:
                assert arg.startswith('--')
                key = arg[2:]

        return kwargs
