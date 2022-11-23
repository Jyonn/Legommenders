def hello(x, *args, **kwargs):
    print('old args', args)
    print('old kwargs', kwargs)

    def decorator(*args, **kwargs):
        print('x', x)
        print('args', args)
        print('kwargs', kwargs)
        return x(*args, **kwargs)
    return decorator


@hello
class A(object):
    def __init__(self, wow):
        self.wow = wow

    def __str__(self):
        return f'<A> wow={self.wow}'


a = A(222)
print(a)

