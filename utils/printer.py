import datetime
import string
import time
from typing import Optional, List

import termcolor

from utils.logger import Logger


"""
red, green, yellow, blue, magenta, cyan, white
"""


class TimePrefix:
    start_time: float

    @classmethod
    def init(cls):
        cls.start_time = time.time()

    @staticmethod
    def div_num(n, base=60):
        return n // base, n % base

    @classmethod
    def call(cls):
        second = int(time.time() - cls.start_time)
        minutes, second = cls.div_num(second)
        hours, minutes = cls.div_num(minutes)
        return '%02d:%02d:%02d' % (hours, minutes, second)


class Bracket:
    DEFAULT = '['
    CLASS = '{'
    METHOD = '<'
    DOT = '.'
    POINT = '·'


class Color:
    RED = 'red'
    GREEN = 'green'
    YELLOW = 'yellow'
    BLUE = 'blue'
    MAGENTA = 'magenta'
    CYAN = 'cyan'
    WHITE = 'white'


class Prefix:
    @classmethod
    def format_bracket(cls, bracket):
        if not bracket:
            bracket = '['

        brackets = ['[]', '{}', '<>']
        for b in brackets:
            if bracket in b:
                return b[0] + '%s' + b[1]

        return bracket + '%s' + bracket

    def __init__(self, prefix, bracket=None, color=None):
        if isinstance(prefix, str):
            prefix = prefix.replace('__', ' ')

        self.prefix = prefix
        self.bracket = self.format_bracket(bracket)
        self.color = color

    def __eq__(self, other: 'Prefix'):
        if not isinstance(other, Prefix):
            return False
        return self.prefix == other.prefix and self.bracket == other.bracket and self.color == other.color

    def get_string(self):
        prefix = self.prefix() if callable(self.prefix) else self.prefix
        return self.bracket % prefix

    def colored_string(self, color=None):
        s = self.get_string()
        return termcolor.colored(s, self.color or color)

    def __str__(self):
        return self.get_string()

    def __repr__(self):
        return str(self)


class Printer:
    depot_k: list
    depot_v: list
    prefix_color: str
    logger: Optional[Logger]

    @classmethod
    def init(cls, prefix_color=None):
        cls.depot_k = []
        cls.depot_v = []
        cls.prefix_color = prefix_color
        cls.logger = None

    @classmethod
    def format_prefix(cls, prefix):
        if isinstance(prefix, Prefix):
            return prefix
        if isinstance(prefix, tuple) or isinstance(prefix, list):
            return Prefix(*prefix)
        if isinstance(prefix, dict):
            return Prefix(**prefix)
        if callable(prefix):
            return Prefix(prefix)
        return Prefix(str(prefix))

    @classmethod
    def format_prefixes(cls, prefixes):
        return [cls.format_prefix(prefix) for prefix in prefixes]

    @classmethod
    def create(cls, prefixes):
        prefixes = cls.format_prefixes(prefixes)

        for k, v in zip(cls.depot_k, cls.depot_v):
            if len(k) != len(prefixes):
                continue

            for p1, p2 in zip(k, prefixes):
                if p1 != p2:
                    break
            else:
                return v

        printer = cls(prefixes=prefixes)
        cls.depot_k.append(prefixes)
        cls.depot_v.append(printer)

        return printer

    def __init__(self, prefixes: list):
        self.prefixes = self.format_prefixes(prefixes)  # type: List[Prefix]

    def __getattr__(self, prefix):
        return self[prefix]

    @staticmethod
    def one_line_prefix(prefix):
        prefix = prefix[:-1]
        e = prefix.rfind('_')
        if e == -1:
            raise ValueError('If not in one-line prefix mode, prefix should not ends with _ char')
        custom = prefix[e+1:]
        prefix = prefix[:e]
        if not custom:
            return prefix

        bracket = None
        bracket_dict = dict(C=Bracket.CLASS, M=Bracket.METHOD, D=Bracket.DEFAULT, P=Bracket.POINT)
        if custom[0] in bracket_dict:
            bracket = bracket_dict[custom[0]]
        if custom[0] in string.ascii_uppercase:
            custom = custom[1:]
        color = custom
        color_dict = dict(r=Color.RED, g=Color.GREEN, b=Color.BLUE, c=Color.CYAN, m=Color.MAGENTA, w=Color.WHITE, y=Color.YELLOW)
        if color in color_dict:
            color = color_dict[color]

        return prefix, bracket, color

    def __getitem__(self, prefix):
        if isinstance(prefix, str):
            if prefix.endswith('_'):
                prefix = self.one_line_prefix(prefix)

        prefixes = [*self.prefixes, prefix]
        return Printer.create(prefixes=prefixes)

    def get_prefix_strings(self):
        return [prefix.get_string() for prefix in self.prefixes]

    def get_colored_prefix_strings(self):
        return [prefix.colored_string(Printer.prefix_color) for prefix in self.prefixes]

    def __call__(self, *args):
        prefix_string = ' '.join(self.get_prefix_strings())
        display_string = ' '.join([str(arg) for arg in args])

        print(*self.get_colored_prefix_strings(), display_string)

        if Printer.logger:
            Printer.logger(' '.join([prefix_string, display_string]))

    def __str__(self):
        return ' '.join(self.get_prefix_strings())

    def __repr__(self):
        return str(self)


TimePrefix.init()
Printer.init()
printer = Printer.create([(TimePrefix.call, Bracket.DEFAULT, Color.GREEN)])
# printer('START TIME:', datetime.datetime.now())

if __name__ == '__main__':
    printer(f'Smart Printer Activated at {datetime.datetime.now()}')
    printer[('wow', '·', Color.CYAN)]('hello')
    printer.wow_Cy_.yes_m_('!')
    printer.wow_Cy_.yes_m_('#')
    printer.wow_Cy_.yes_m_.hungry('I am so hungry')
    # printer.c('x')
    # printer.c('y')
    print(Printer.depot_k)
