# coding=utf-8
from __future__ import unicode_literals, absolute_import

import json
from textwrap import dedent

from juicer.operation import Operation
from juicer.spark.vis_operation import get_caipirinha_config


class FairnessEvaluationOperation(Operation):
    SENSITIVE_ATTRIBUTE_PARAM = 'attributes'
    LABEL_ATTRIBUTE_PARAM = 'label'
    SCORE_ATTRIBUTE_PARAM = 'score'

    TYPE_PARAM = 'type'
    TAU_PARAM = 'tau'
    BASELINE_PARAM = 'baseline'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.tau = parameters.get(self.TAU_PARAM, 0.8)

        self.sensitive = parameters.get(self.SENSITIVE_ATTRIBUTE_PARAM, [])
        if not self.sensitive:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.SENSITIVE_ATTRIBUTE_PARAM, self.__class__))

        self.baseline = parameters.get(self.BASELINE_PARAM, None)
        if not self.sensitive:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.BASELINE_PARAM, self.__class__))

        self.label = (parameters.get(
            self.LABEL_ATTRIBUTE_PARAM, ['label']) or ['label'])[0]
        self.score = (parameters.get(
            self.LABEL_ATTRIBUTE_PARAM, ['score']) or ['score'])[0]
        self.type = parameters.get(self.TYPE_PARAM, 'aequitas')

        self.output = self.named_outputs.get(
            'output data', 'out_task_{}'.format(self.order))
        self.input = self.named_inputs.get('input data')

        self.has_code = all([
            len(self.named_inputs) == 1,
            any([len(self.named_outputs) == 1, self.contains_results()])])

    def generate_code(self):
        display_image = self.parameters['task']['forms'].get(
            'display_image', {'value': 1}).get('value', 1) in (1, '1')
        code = dedent("""

            from juicer.service import caipirinha_service
            from juicer.spark.vis_operation import HtmlVisualizationModel
            from juicer.spark.ext import FairnessEvaluatorTransformer
            from juicer.spark.reports import FairnessBiasReport

            baseline = '{baseline}'

            sensitive_column_dt = {input}.schema[str('{sensitive}')].dataType
            if isinstance(sensitive_column_dt, types.FractionalType):
                baseline = float(baseline)
            elif isinstance(sensitive_column_dt, types.IntegralType):
                baseline = int(baseline)
            elif not isinstance(sensitive_column_dt, types.StringType):
                raise ValueError(_('Invalid column type: {{}}').format(
                sensitive_column_dt))

            evaluator = FairnessEvaluatorTransformer(
                sensitiveColumn='{sensitive}', labelColumn='{label}',
                   baselineValue=baseline, tau={tau})
            {out} = evaluator.transform({input})
            display_image = {display_image}
            if display_image:
                visualization = {{
                    'job_id': '{job_id}', 'task_id': '{task_id}',
                    'title': '{title}',
                    'type': {{'id': 1, 'name': 'HTML'}},
                    'model': HtmlVisualizationModel(title='{title}'),
                    'data': json.dumps({{
                        'html': FairnessBiasReport({out},
                            '{sensitive}', baseline).generate(),
                        'xhtml': '''
                            <a href="" target="_blank">
                            {title} ({open})
                            </a>'''
                    }}),
                }}
                {config}
                # emit_event(
                #             'update task', status='COMPLETED',
                #             identifier='{task_id}',
                #             message=base64.b64encode(fig_file.getvalue()),
                #             type='IMAGE', title='{title}',
                #             task={{'id': '{task_id}'}},
                #             operation={{'id': {operation_id}}},
                #             operation_id={operation_id})

                caipirinha_service.new_visualization(
                    config,
                    {user},
                    {workflow_id}, {job_id}, '{task_id}',
                    visualization, emit_event)
        """.format(sensitive=self.sensitive[0], label=self.label,
                   baseline=self.baseline, tau=self.tau,
                   out=self.output, input=self.input,
                   display_image=display_image,
                   task_id=self.parameters['task_id'],
                   operation_id=self.parameters['operation_id'],
                   job_id=self.parameters['job_id'],
                   user=self.parameters['user'],
                   workflow_id=self.parameters['workflow_id'],
                   config=get_caipirinha_config(self.config, indentation=16),
                   title=_('Bias/Fairness Report'),
                   open=_('click to open')))
        return code
