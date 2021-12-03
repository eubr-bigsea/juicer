from textwrap import dedent, indent

import re
import json
from juicer.operation import Operation
from juicer.spark.vis_operation import (
    VisualizationMethodOperation, HtmlVisualizationModel)
from gettext import gettext

class ValidationOperation(Operation):
    PARAM_CATEGORY = 'category'
    PARAM_VALIDATION = 'validation'
    PARAM_ALLOW_CROSS_TYPE_COMPARISONS = 'allow_cross_type_comparisons'
    PARAM_ALLOW_RELATIVE_ERROR = 'allow_relative_error'
    PARAM_BINS_A = 'bins_A'
    PARAM_BINS_B = 'bins_B'
    PARAM_BOOTSTRAP_SAMPLE_SIZE = 'bootstrap_sample_size'
    PARAM_BOOTSTRAP_SAMPLES = 'bootstrap_samples'
    PARAM_BUCKETIZE_DATA = 'bucketize_data'
    PARAM_COLUMNS = 'columns'
    PARAM_COLUMN_A = 'column_A'
    PARAM_COLUMN_B = 'column_B'
    PARAM_COLUMN_INDEX = 'column_index'
    PARAM_COLUMN_LIST = 'column_list'
    PARAM_COLUMN_SET = 'column_set'
    PARAM_DISTRIBUTION = 'distribution'
    PARAM_EXACT_MATCH = 'exact_match'
    PARAM_IGNORE_ROW_IF = 'ignore_row_if'
    PARAM_INTERNAL_WEIGHT_HOLDOUT = 'internal_weight_holdout'
    PARAM_JSON_SCHEMA = 'json_schema'
    PARAM_MATCH_ON = 'match_on'
    PARAM_MAX_VALUE = 'max_value'
    PARAM_MIN_VALUE = 'min_value'
    PARAM_MOSTLY = 'mostly'
    PARAM_N_BINS_A = 'n_bins_A'
    PARAM_N_BINS_B = 'n_bins_B'
    PARAM_OR_EQUAL = 'or_equal'
    PARAM_OUTPUT_STRFTIME_FORMAT = 'output_strftime_format'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

    
        self.has_code = (len(self.named_inputs) == 1 and 
            len(self.named_outputs) > 0)
        for p in [self.PARAM_CATEGORY, self.PARAM_VALIDATION]:
            if p not in parameters:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                   .format(p, self.__class__))
        self.parameters = parameters
        self.category = parameters.get(self.PARAM_CATEGORY)
        self.validation = 'expect_' + parameters.get(self.PARAM_VALIDATION)

        self.transpiler_utils.add_import('import great_expectations as ge')
        self.input = named_inputs.get('input data')
        self.output = named_outputs.get('output results')

    def _format_params(self, params):
        return '\n' + indent(',\n'.join([f'{k}={v}' for (k, v) in params.items()]), 
            12 * ' ')

    def generate_code(self):
        import great_expectations as ge
        import inspect
        if not hasattr(ge.dataset.dataset.Dataset, self.validation):
            raise ValueError(
                _("Invalid validation name: {}").format(self.validation))

        method = getattr(ge.dataset.dataset.Dataset, self.validation)
        params = {}
        # Last four are 'result_format', 'include_config', 'catch_exceptions', 'meta'
        param_names = list(inspect.signature(method).parameters.keys())[1:-4]
        for p in param_names:
            if p in self.parameters:
                params[p] = self.parameters[p]

        params.update({'result_format': "'SUMMARY'", 'include_config': False, 
            'catch_exceptions': True, 'meta': False})
        columns = self.parameters.get(self.PARAM_COLUMNS, []) 
        multiple_columns = len(columns) > 1

        code = [
            f'df_ge = ge.dataset.SparkDFDataset({self.input})',
            f'result = []',

        ]
        if multiple_columns:
            for column in columns:
                params['column'] = f"'{column}'"
                kwa = self._format_params(params)
                code.append(
                    f'result.append(\n    df_ge.{self.validation}({kwa}))')
        else:
            params['column'] = f"'{columns[0]}'"
            kwa = self._format_params(params)
            code.append(
                f'result.append(\n    df_ge.{self.validation}({kwa}))')

        code.append(f'{self.output} = df_ge ')
        return dedent('\n'.join(code))


class ValidationReportOperation(Operation):
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.has_code = len(named_inputs) > 0
        self.transpiler_utils.add_import("from juicer.service import caipirinha_service")
        self.transpiler_utils.add_import(
            "from great_expectations.render.renderer import ValidationResultsPageRenderer")
        self.transpiler_utils.add_import(
            "from great_expectations.render.view import DefaultJinjaPageView")
        
    def generate_code(self):
        title = gettext('Validation report')
        config = self.config['juicer']['services']['caipirinha']
        params = self.parameters
        inputs = self.named_inputs['input results']
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        df_ge = inputs[0]
        
        code = f"""
        validation_result = {df_ge}.validate()
        document_model = ValidationResultsPageRenderer().render(validation_result)
        html = DefaultJinjaPageView().render(document_model)
        visualization = {{
            'job_id': '{params['job_id']}',
            'task_id': '{params['task_id']}',
            'title': '{title}',
            'type': {{'id': 1}}, #HTML
            'data': json.dumps({{'data': html}}),
        }}
        caipirinha_service.new_visualization(
            {config},
            {params['user']},
            {params['workflow_id']}, 
            {params['job_id']}, 
            '{params['task_id']}',
            visualization, emit_event)
        """
        return dedent(code)

