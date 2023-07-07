# make a dict slicable
from collections import OrderedDict


class SliceDict(dict):
    def __getitem__(self, item):
        if not isinstance(item, slice):
            return super(SliceDict, self).__getitem__(item)

        slice_dict = SliceDict()
        for k, v in self.items():
            slice_dict[k] = v[item]
        return slice_dict


class SliceOrderedDict(OrderedDict):
    def __getitem__(self, item):
        if not isinstance(item, slice):
            return super(SliceOrderedDict, self).__getitem__(item)

        slice_dict = SliceOrderedDict()
        for k, v in self.items():
            slice_dict[k] = v[item]
        return slice_dict


if __name__ == '__main__':
    d = SliceDict(
        a=[1, 2, 3],
        b=[4, 5, 6],
        c=[7, 8, 9],
    )
    print(d[1:])

    d = dict(a=1, b=2)
    e = dict(d)
    print(e)

