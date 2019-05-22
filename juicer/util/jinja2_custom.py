# -*- coding: utf-8 -*-

from jinja2 import nodes
from jinja2.ext import Extension
import autopep8


class AutoPep8Extension(Extension):
    # a set of names that trigger the extension.
    tags = {'autopep8'}

    def __init__(self, environment):
        super(AutoPep8Extension, self).__init__(environment)
        # add the defaults to the environment
        environment.extend()

    def parse(self, parser):
        lineno = next(parser.stream).lineno

        body = parser.parse_statements(['name:endautopep8'], drop_needle=True)
        args = []
        result = nodes.CallBlock(self.call_method('_format_support', args),
                                 [], [], body).set_lineno(lineno)
        return result

    @staticmethod
    def _format_support(caller):
        return autopep8.fix_code(caller(), options={'aggressive': 1})