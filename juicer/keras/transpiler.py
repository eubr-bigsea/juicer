# -*- coding: utf-8 -*-


import os
from juicer import operation
from juicer.keras import merge_layers
from juicer.keras import core_layers
from juicer.keras import convolutional_layers
from juicer.keras import pooling_layers
from juicer.keras import recurrent_layers
from juicer.keras import normalization_layers
from juicer.keras import pre_trained_layers
from juicer.keras import advanced_layers
from juicer.keras import model
from juicer.keras import preprocessing
from juicer.keras import input_output_data
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
            'add': merge_layers.Add,
            'average': merge_layers.Average,
            'average-pooling-1d': pooling_layers.AveragePooling1D,
            'average-pooling-2d': pooling_layers.AveragePooling2D,
            'average-pooling-3d': pooling_layers.AveragePooling3D,
            'batch-normalization': normalization_layers.BatchNormalization,
            'concatenate': merge_layers.Concatenate,
            #'conv-lstm-2D': core_layers.ConvLSTM2D,
            'convolution-1d': convolutional_layers.Convolution1D,
            'convolution-2d': convolutional_layers.Convolution2D,
            'conv-2d-transpose': convolutional_layers.Conv2DTranspose,
            'convolution-3d': convolutional_layers.Convolution3D,
            'conv-3d-transpose': convolutional_layers.Conv3DTranspose,
            'cropping-1d': convolutional_layers.Cropping1D,
            'cropping-2d': convolutional_layers.Cropping2D,
            'cropping-3d': convolutional_layers.Cropping3D,
            #'cu-dnn-gru': core_layers.CUDNNGRU,
            #'cu-dnn-lstm': core_layers.CUDNNLSTM,
            'depth-wise-conv-2d': convolutional_layers.DepthwiseConv2D,
            'dense': core_layers.Dense,
            'dot': merge_layers.Dot,
            'dropout': core_layers.Dropout,
            'evaluate-model': model.EvaluateModel,
            'file-reader': input_output_data.FileReader,
            'fit-generator': model.FitGenerator,
            'flatten': core_layers.Flatten,
            'global-average-pooling-1d': pooling_layers.GlobalAveragePooling1D,
            'global-average-pooling-2d': pooling_layers.GlobalAveragePooling2D,
            'global-average-pooling-3d': pooling_layers.GlobalAveragePooling3D,
            'global-max-pooling-1d': pooling_layers.GlobalMaxPooling1D,
            'global-max-pooling-2d': pooling_layers.GlobalMaxPooling2D,
            'global-max-pooling-3d': pooling_layers.GlobalMaxPooling3D,
            #'GRU': core_layers.GRU,
            #'GRUCell': core_layers.GRUCell,
            #'hyperparameters': core_layers.Hyperparameters,
            'image-generator': preprocessing.ImageGenerator,
            'image-reader': input_output_data.ImageReader,
            'inception-v3': pre_trained_layers.InceptionV3,
            'input': core_layers.Input,
            'lambda': core_layers.Lambda,
            #'loss': core_layers.Loss,
            #'loss-operation': core_layers.LossOperation,
            'load': model.Load,
            'lstm': recurrent_layers.LSTM,
            #'lstm-cell': core_layers.LSTMCell,
            'masking': core_layers.Masking,
            'max-pooling-1d': pooling_layers.MaxPooling1D,
            'max-pooling-2d': pooling_layers.MaxPooling2D,
            'max-pooling-3d': pooling_layers.MaxPooling3D,
            'maximum': merge_layers.Maximum,
            'minimum': merge_layers.Minimum,
            'model': model.Model,
            #'model-generator': core_layers.ModelGenerator,
            'multiply': merge_layers.Multiply,
            #'optimizer': core_layers.OptimizerOperation,
            #'output': core_layers.OutputOperation,
            'permute': core_layers.Permute,
            'predict': model.Predict,
            'python-code': advanced_layers.PythonCode,
            'repeatVector': core_layers.RepeatVector,
            'reshape': core_layers.Reshape,
            #'rnn':core_layers.RNN,
            'separable-conv-1d': convolutional_layers.SeparableConv1D,
            'separable-conv-2d': convolutional_layers.SeparableConv2D,
            'sequence-reader': input_output_data.SequenceReader,
            'sequence-generator': preprocessing.SequenceGenerator,
            'simple-rnn': recurrent_layers.SimpleRNN,
            #'simple-rnn-cell': core_layers.SimpleRNNCell,
            'spatialDropout1D': core_layers.SpatialDropout1D,
            'spatialDropout2D': core_layers.SpatialDropout2D,
            'spatialDropout3D': core_layers.SpatialDropout3D,
            'subtract': merge_layers.Subtract,
            'up-sampling-1d': convolutional_layers.UpSampling1D,
            'up-sampling-2d': convolutional_layers.UpSampling2D,
            'up-sampling-3d': convolutional_layers.UpSampling3D,
            'vgg-16': pre_trained_layers.VGG16,
            'video-reader': input_output_data.VideoReader,
            'video-generator': preprocessing.VideoGenerator,
            'zero-padding-1d': convolutional_layers.ZeroPadding1D,
            'zero-padding-2d': convolutional_layers.ZeroPadding2D,
            'zero-padding-3d': convolutional_layers.ZeroPadding3D,
        }

        self.operations = {}
        for ops in [core_ops, ]:
            self.operations.update(ops)
