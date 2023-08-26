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

        for key, value in kwargs.items():
            if value == 'null':
                kwargs[key] = None
            elif value.isdigit():
                kwargs[key] = int(value)
            elif value.lower() == 'true':
                kwargs[key] = True
            elif value.lower() == 'false':
                kwargs[key] = False
            else:
                try:
                    kwargs[key] = float(value)
                except ValueError:
                    pass
        return kwargs
