# coding=utf-8
class JuicerException(Exception):
    def __init__(self, *args, **kwargs):
        if 'code' in kwargs:
            self.code = kwargs.pop('code')
        else:
            self.code = 501
        Exception.__init__(self, *args, **kwargs)


class InvalidGeneratedCode(JuicerException):
    pass
