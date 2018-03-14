# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from collections import namedtuple
from textwrap import dedent

from juicer.operation import Operation

ScalerNameAndParameters = namedtuple("ScalerNameAndParameters",
                                     "name, parameters, metrics")


class ScalerOperation(Operation):
    """
    Base class for operations to scale data.
    """

    ATTRIBUTE_PARAM = 'attribute'
    ALIAS_PARAM = 'alias'

    __slots__ = ('attribute', 'scaled_attr', 'model')

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.ATTRIBUTE_PARAM in parameters:
            self.attribute = parameters.get(self.ATTRIBUTE_PARAM)[0]
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTE_PARAM, self.__class__))
        self.scaled_attr = parameters.get(self.ALIAS_PARAM,
                                          'scaled_{}'.format(self.order))

        self.has_code = 'input data' in self.named_inputs
        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.model = self.named_outputs.get('transformation model',
                                            'model_task_{}'.format(self.order))

    def _get_scaler_algorithm_and_parameters(self):
        raise NotImplementedError(_('Must be implemented in children classes'))

    def generate_code(self):
        name_and_params = self._get_scaler_algorithm_and_parameters()
        name_and_params.parameters['inputCol'] = self.attribute
        name_and_params.parameters['outputCol'] = self.scaled_attr

        input_data = self.named_inputs['input data']

        code = dedent("""
            params = {params}
            metrics = {metrics}
            scaler = {scaler_impl}(**params)
            {model} = scaler.fit({input})

            # Lazy execution in case of sampling the data in UI
            def call_transform(df):
                return {model}.transform(df)
            {output} = dataframe_util.LazySparkTransformationDataframe(
                {model}, {input}, call_transform)
            if metrics:
                from juicer.spark.reports import SimpleTableReport
                headers = ['Metric', 'Value']
                rows = [[m, getattr({model}, m)] for m in metrics]

                content = SimpleTableReport(
                        'table table-striped table-bordered table-sm',
                        headers, rows)

                emit_event(
                    'update task', status='COMPLETED',
                    identifier='{task_id}',
                    message=content.generate(),
                    type='HTML', title='{title}',
                    task={{'id': '{task_id}'}},
                    operation={{'id': {operation_id}}},
                    operation_id={operation_id})
        """.format(
            params=name_and_params.parameters,  # str representation is ok here
            scaler_impl=name_and_params.name,
            input=input_data,
            output=self.output,
            model=self.model,
            metrics=name_and_params.metrics,
            task_id=self.parameters['task_id'],
            operation_id=self.parameters['operation_id'],
            title="Metrics for task",
        ))

        return code


class StandardScalerOperation(ScalerOperation):
    """
    Standardizes features by removing the mean and scaling to unit variance
    using column summary statistics on the samples in the training set.
    """

    WITH_MEAN_PARAM = 'with_mean'
    WITH_STD_PARAM = 'with_std'
    __slots__ = ('with_std', 'with_mean')

    def __init__(self, parameters, named_inputs, named_outputs):
        ScalerOperation.__init__(self, parameters, named_inputs, named_outputs)
        self.with_mean = parameters.get(
            self.WITH_MEAN_PARAM, False) in ['1', 1, True]
        self.with_std = parameters.get(
            self.WITH_STD_PARAM, True) in ['1', 1, True]

    def _get_scaler_algorithm_and_parameters(self):
        return ScalerNameAndParameters(
            'StandardScaler',
            {'withStd': self.with_std, 'withMean': self.with_mean},
            ['mean', 'std', ]
        )


class MaxAbsScalerOperation(ScalerOperation):
    """
    Rescale each feature individually to range [-1, 1] by dividing through the
    largest maximum absolute value in each feature. It does not shift/center the
    data, and thus does not destroy any sparsity.
    """
    __slots__ = []

    def __init__(self, parameters, named_inputs, named_outputs):
        ScalerOperation.__init__(self, parameters, named_inputs, named_outputs)

    def _get_scaler_algorithm_and_parameters(self):
        return ScalerNameAndParameters('MaxAbsScaler', {}, [])


class MinMaxScalerOperation(ScalerOperation):
    """
    Rescale each feature individually to range [-1, 1] by dividing through the
    largest maximum absolute value in each feature. It does not shift/center the
    data, and thus does not destroy any sparsity.
    """
    __slots__ = ['min', 'max']
    MIN_PARAM = 'min'
    MAX_PARAM = 'max'

    def __init__(self, parameters, named_inputs, named_outputs):
        ScalerOperation.__init__(self, parameters, named_inputs, named_outputs)
        self.min = float(parameters.get(self.MIN_PARAM, 0.0))
        self.max = float(parameters.get(self.MAX_PARAM, 1.0))

    def _get_scaler_algorithm_and_parameters(self):
        return ScalerNameAndParameters(
            'MinMaxScaler', {'min': self.min, 'max': self.max}, ['min', 'max'])
