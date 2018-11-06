# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import juicer.keras.core_operations as core_operations
import os
from juicer import operation
from juicer.transpiler import Transpiler


class KerasTranspiler(Transpiler):
    """
    Convert Lemonade workflow representation (JSON) into code to be run in
    Keras.
    """

    def __init__(self, configuration, slug_to_op_id=None, port_id_to_port=None):
        super(KerasTranspiler, self).__init__(
            configuration, os.path.abspath(os.path.dirname(__file__)),
            slug_to_op_id, port_id_to_port)

    # noinspection SpellCheckingInspection
    def _assign_operations(self):
        core_ops = {
            'dense': core_operations.DenseOperation,
            'dropout': core_operations.DropoutOperation,
            'flatten': core_operations.FlattenOperation,
            'optimizer': core_operations.OptimizerOperation,
            'loss': core_operations.LossOperation,
            'input': core_operations.InputOperation,
            'output': core_operations.OutputOperation,
            'activation': core_operations.ActivationOperation,
            'reshape': core_operations.ReshapeOperation,
            'permute': core_operations.PermuteOperation,
            'repeatVector': core_operations.RepeatVectorOperation,
            'lambda': core_operations.LambdaOperation,
            'activityRegularization': core_operations.ActivityRegularizationOperation,
            'masking': core_operations.MaskingOperation,
        }

        self.operations = {}
        for ops in [core_ops, ]:
            self.operations.update(ops)
