# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import juicer.keras.core_layers as core_layers, convolutional_layers
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
            'activation': core_layers.Activation,
            'activityRegularization': core_layers.ActivityRegularization,
            'add': core_layers.Add,
            'average': core_layers.Average,
            'average-pooling-1d': core_layers.AveragePooling1D,
            'average-pooling-2d': core_layers.AveragePooling2D,
            'average-pooling-3d': core_layers.AveragePooling3D,
            'batch-normalization': core_layers.BatchNormalization,
            'concatenate': core_layers.Concatenate,
            #'conv-lstm-2D': core_layers.ConvLSTM2D,
            'convolution-1d': convolutional_layers.Convolution1D,
            'convolution-2d': convolutional_layers.Convolution2D,
            'conv-2d-transpose': convolutional_layers.Conv2DTranspose,
            'conv-3d': convolutional_layers.Conv3D,
            'conv-3d-transpose': convolutional_layers.Conv3DTranspose,
            'cropping-1d': convolutional_layers.Cropping1D,
            'cropping-2d': convolutional_layers.Cropping2D,
            'cropping-3d': convolutional_layers.Cropping3D,
            #'cu-dnn-gru': core_layers.CUDNNGRU,
            #'cu-dnn-lstm': core_layers.CUDNNLSTM,
            'depth-wise-conv-2d': convolutional_layers.DepthwiseConv2D,
            'dense': core_layers.Dense,
            'dot': core_layers.Dot,
            'dropout': core_layers.Dropout,
            'flatten': core_layers.Flatten,
            'global-average-pooling-1d': core_layers.GlobalAveragePooling1D,
            'global-average-pooling-2d': core_layers.GlobalAveragePooling2D,
            'global-average-pooling-3d': core_layers.GlobalAveragePooling3D,
            'global-max-pooling-1d': core_layers.GlobalMaxPooling1D,
            'global-max-pooling-2d': core_layers.GlobalMaxPooling2D,
            'global-max-pooling-3d': core_layers.GlobalMaxPooling3D,
            #'GRU': core_layers.GRU,
            #'GRUCell': core_layers.GRUCell,
            #'hyperparameters': core_layers.Hyperparameters,
            'image-generator': core_layers.ImageGenerator,
            'image-reader': core_layers.ImageReader,
            'inception-v3': core_layers.InceptionV3,
            'input': core_layers.Input,
            'lambda': core_layers.Lambda,
            #'loss': core_layers.Loss,
            #'loss-operation': core_layers.LossOperation,
            'lstm': core_layers.LSTM,
            #'lstm-cell': core_layers.LSTMCell,
            'masking': core_layers.Masking,
            'max-pooling-1d': core_layers.MaxPooling1D,
            'max-pooling-2d': core_layers.MaxPooling2D,
            'max-pooling-3d': core_layers.MaxPooling3D,
            'maximum': core_layers.Maximum,
            'minimum': core_layers.Minimum,
            'model': core_layers.ModelGenerator,
            #'model-generator': core_layers.ModelGenerator,
            'multiply': core_layers.Multiply,
            #'optimizer': core_layers.OptimizerOperation,
            #'output': core_layers.OutputOperation,
            'permute': core_layers.Permute,
            'python-code': core_layers.PythonCode,
            'repeatVector': core_layers.RepeatVector,
            'reshape': core_layers.Reshape,
            #'rnn':core_layers.RNN,
            'separable-conv-1d': convolutional_layers.SeparableConv1D,
            'separable-conv-2d': convolutional_layers.SeparableConv2D,
            'simple-rnn': core_layers.SimpleRNN,
            #'simple-rnn-cell': core_layers.SimpleRNNCell,
            'spatialDropout1D': core_layers.SpatialDropout1D,
            'spatialDropout2D': core_layers.SpatialDropout2D,
            'spatialDropout3D': core_layers.SpatialDropout3D,
            'subtract': core_layers.Subtract,
            #'up-sampling-1d': core_layers.UpSampling1D,
            #'up-sampling-2d': core_layers.UpSampling2D,
            #'up-sampling-3d': core_layers.UpSampling3D,
            'vgg-16': core_layers.VGG16,
            'video-reader': core_layers.VideoReader,
            'video-generator': core_layers.VideoGenerator,
            #'zero-padding-1d': core_layersZeroPadding1D,
            #'zero-padding-2d': core_layersZeroPadding2D,
            'zero-padding-3d': convolutional_layers.ZeroPadding3D,
        }

        self.operations = {}
        for ops in [core_ops, ]:
            self.operations.update(ops)
