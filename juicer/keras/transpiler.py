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
            'activation': core_operations.ActivationOperation,
            'activityRegularization': core_operations.ActivityRegularizationOperation,
            'add': core_operations.Add,
            'average': core_operations.Average,
            'average-pooling-1d': core_operations.AveragePooling1D,
            'average-pooling-2d': core_operations.AveragePooling2D,
            'average-pooling-3d': core_operations.AveragePooling3D,
            'batch-normalization': core_operations.BatchNormalization,
            'concatenate': core_operations.Concatenate,
            #'conv-lstm-2D': core_operations.ConvLSTM2D,
            #'convolution-1d': core_operations.Convolution1D,
            'convolution-2d': core_operations.Convolution2D,
            #'conv-2d-transpose': core_operationsConv2DTranspose,
            #'conv-3d': core_operationsConv3D,
            #'conv-3d-transpose': core_operationsConv3DTranspose,
            #'cropping-1d': core_operations.Cropping1D,
            #'cropping-2d': core_operations.Cropping2D,
            #'cropping-3d': core_operations.Cropping3D,
            #'cu-dnn-gru': core_operations.CUDNNGRU,
            #'cu-dnn-lstm': core_operations.CUDNNLSTM,
            #'depth-wise-conv-2d': core_operationsDepthWiseConv2D,
            'dense': core_operations.DenseOperation,
            'dot': core_operations.Dot,
            'dropout': core_operations.DropoutOperation,
            'flatten': core_operations.FlattenOperation,
            'global-average-pooling-1d': core_operations.GlobalAveragePooling1D,
            'global-average-pooling-2d': core_operations.GlobalAveragePooling2D,
            'global-average-pooling-3d': core_operations.GlobalAveragePooling3D,
            'global-max-pooling-1d': core_operations.GlobalMaxPooling1D,
            'global-max-pooling-2d': core_operations.GlobalMaxPooling2D,
            'global-max-pooling-3d': core_operations.GlobalMaxPooling3D,
            #'GRU': core_operations.GRU,
            #'GRUCell': core_operations.GRUCell,
            #'hyperparameters': core_operations.Hyperparameters,
            'image-generator': core_operations.ImageGenerator,
            'image-reader': core_operations.ImageReader,
            'inception-v3': core_operations.InceptionV3,
            'input': core_operations.InputOperation,
            'lambda': core_operations.LambdaOperation,
            #'loss': core_operations.Loss,
            #'loss-operation': core_operations.LossOperation,
            'lstm': core_operations.LSTM,
            #'lstm-cell': core_operations.LSTMCell,
            'masking': core_operations.MaskingOperation,
            'max-pooling-1d': core_operations.MaxPooling1D,
            'max-pooling-2d': core_operations.MaxPooling2D,
            'max-pooling-3d': core_operations.MaxPooling3D,
            'maximum': core_operations.Maximum,
            'minimum': core_operations.Minimum,
            'model': core_operations.ModelGenerator,
            #'model-generator': core_operations.ModelGenerator,
            'multiply': core_operations.Multiply,
            #'optimizer': core_operations.OptimizerOperation,
            #'output': core_operations.OutputOperation,
            'permute': core_operations.PermuteOperation,
            'python-code': core_operations.PythonCode,
            'repeatVector': core_operations.RepeatVectorOperation,
            'reshape': core_operations.ReshapeOperation,
            #'rnn':core_operations.RNN,
            #'separable-conv-1d': core_operations.SeparableConv1D,
            #'separable-conv-2d': core_operations.SeparableConv2D,
            'simple-rnn': core_operations.SimpleRNN,
            #'simple-rnn-cell': core_operations.SimpleRNNCell,
            'spatialDropout1D': core_operations.SpatialDropout1DOperation,
            'spatialDropout2D': core_operations.SpatialDropout2DOperation,
            'spatialDropout3D': core_operations.SpatialDropout3DOperation,
            'subtract': core_operations.Subtract,
            #'up-sampling-1d': core_operations.UpSampling1D,
            #'up-sampling-2d': core_operations.UpSampling2D,
            #'up-sampling-3d': core_operations.UpSampling3D,
            'vgg-16': core_operations.VGG16,
            #'zero-padding-1d': core_operationsZeroPadding1D,
            #'zero-padding-2d': core_operationsZeroPadding2D,
            #'zero-padding-3d': core_operationsZeroPadding3D,
        }

        self.operations = {}
        for ops in [core_ops, ]:
            self.operations.update(ops)
