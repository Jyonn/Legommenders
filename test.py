def hello(x, *args, **kwargs):
    print(x)
    print(args)
    print(kwargs)
    return x


class A:
    @hello
    def __init__(self, wow):
        pass


print("??")
# A(222)
