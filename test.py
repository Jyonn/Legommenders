import torch


# def hello(x, *args, **kwargs):
#     print('old args', args)
#     print('old kwargs', kwargs)
#
#     def decorator(*args, **kwargs):
#         print('x', x)
#         print('args', args)
#         print('kwargs', kwargs)
#         return x(*args, **kwargs)
#     return decorator
#
#
# @hello
# class A(object):
#     def __init__(self, wow):
#         self.wow = wow
#
#     def __str__(self):
#         return f'<A> wow={self.wow}'
#
#
# a = A(222)
# print(a)
#
#
# class B:
#     pass
#
#
# b = B()
#
# print(B)
# print(b)

from torch.utils.data.dataloader import default_collate

collate_fn = torch.utils.data.dataloader.default_collate

print(collate_fn([
    dict(x=torch.tensor([1, 2]), y=torch.tensor([[5, 5, 6], [8, 8, 7]])),
    dict(x=torch.tensor([1, 2]), y=torch.tensor([[5, 5, 6], [8, 8, 7]])),
]))
