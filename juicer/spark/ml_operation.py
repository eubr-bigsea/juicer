# coding=utf-8
import logging

from juicer.spark.operation import Operation

log = logging.getLogger()
log.setLevel(logging.DEBUG)


class SvmClassification(Operation):
    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        self.parameters = parameters
        self.has_code = False


class EvaluateModel(Operation):
    def __init__(self, parameters, inputs, outputs):
        Operation.__init__(self, inputs, outputs)
        self.parameters = parameters
        self.has_code = False