# -*- coding: utf-8 -*-
import imp


class StringImporter(object):
    """ Imports Python packages from a string.
    See http://stackoverflow.com/a/14192708/1646932
    """

    def __init__(self):
        self._modules = {}

    def add_or_update_module(self, module, code):
        self._modules[module] = code
        return self.load_module(module)

    # noinspection PyUnusedLocal
    def find_module(self, fullname, path):
        if fullname in self._modules.keys():
            return self._modules[fullname]
        return None

    def load_module(self, fullname):
        if fullname not in self._modules.keys():
            raise ImportError(fullname)

        new_module = imp.new_module(fullname)
        exec self._modules[fullname] in new_module.__dict__
        return new_module
