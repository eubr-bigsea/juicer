# coding=utf-8
from gettext import gettext
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
    COLUMNS = {
        'EP': 'pred_pos_ratio_k_parity',
        'PP': 'pred_pos_ratio_g_parity',
        'FPRP': 'fpr_parity',
        'FDRP': 'fdr_parity',
        'FNRP': 'fnr_parity',
        'FORP': 'for_parity'
    }

    VALID_TYPES = {
        'EP': gettext('Equal Parity'),
        'PP': gettext('Proportional Parity'),
        'FPRP': gettext('False-Positive Parity Rate'),
        'FDRP': gettext('False-Discovery Parity Rate'),
        'FNRP': gettext('False-Negative Parity Rate'),
        'FORP': gettext('False-Omission Parity Rate'),
    }

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        self.tau = parameters.get(self.TAU_PARAM, 0.8)
        
        self.sensitive = parameters.get(self.SENSITIVE_ATTRIBUTE_PARAM, [])
        if not self.sensitive:
            raise ValueError(
                gettext("Parameter '{}' must be informed for task {}").format(
                    self.SENSITIVE_ATTRIBUTE_PARAM, self.__class__))

        self.baseline = parameters.get(self.BASELINE_PARAM, None)
        if not self.sensitive:
            raise ValueError(
                gettext("Parameter '{}' must be informed for task {}").format(
                    self.BASELINE_PARAM, self.__class__))

        self.label = (parameters.get(
            self.LABEL_ATTRIBUTE_PARAM, ['label']) or ['label'])[0]
        self.score = (parameters.get(
            self.SCORE_ATTRIBUTE_PARAM, ['score']) or ['score'])[0]
        #self.type = parameters.get(self.TYPE_PARAM, 'EP')
        self.type = 'EP'

        #import pdb; pdb.set_trace();
        if self.type not in self.VALID_TYPES.keys():
            raise ValueError(
                gettext('Parameter {} must be one of these: {}').format(
                    gettext('type'), ','.join(self.VALID_TYPES.keys())
                )
            )
        self.fairness_metric = self.VALID_TYPES[self.type]
        self.column_name = self.COLUMNS[self.type]

        self.output = self.named_outputs.get(
            'output data', 'out_task_{}'.format(self.order))
        self.input = self.named_inputs.get('input data')

        self.has_code = True
        self.supports_cache = False

    def generate_code(self):
        display_text = self.parameters['task']['forms'].get(
            'display_text', {'value': 1}).get('value', 1) in (1, '1')
        code = dedent("""

                from juicer.spark.vis_operation import HtmlVisualizationModel 
                from juicer.spark.ext import FairnessEvaluatorSql
                from juicer.spark.reports import FairnessBiasReport
                from juicer.spark.reports import SimpleTableReport

                baseline = '{baseline}'

                sensitive_column_dt = {input}.schema[str('{sensitive}')].dataType
                if isinstance(sensitive_column_dt, types.FractionalType):
                    baseline = float(baseline)
                elif isinstance(sensitive_column_dt, types.IntegralType):
                    baseline = int(baseline)
                elif not isinstance(sensitive_column_dt, types.StringType):
                    raise ValueError(gettext('Invalid column type: {{}}').format(
                    sensitive_column_dt))

                #Hardcode - Update    
                evaluator = FairnessEvaluatorSql(sensitive_column='race', score_column='score', 
                                                 label_column='label_value', baseline_column='Caucasian', 
                                                 range_column=[0.8,1.25], type_fairness_sql='list_all_groups_and_metrics',
                                                 percentage_group_size=10, type_disparity='disparity_by_group'  
                                                )
                {input}.createOrReplaceTempView(evaluator.TABLE)
                sql_result = spark_session.sql(evaluator.get_fairness_sql())
                {out} = sql_result

                #{out} = evaluator.transform({input})
                #evaluator = FairnessEvaluatorTransformer(
                #    sensitiveColumn='{sensitive}', labelColumn='{label}',
                #       baselineValue=str(baseline), tau={tau},
                #       scoreColumn='{score}')
                #{out} = evaluator.transform({input})
                display_text = {display_text}


                headers = {headers}
                #rows = {out}.select('{sensitive}' , '{column_name}',
                #    functions.round('{score_column_name}', 2)).collect()

                #Hardcode - Update    
                #rows = out_task_1.select('race', 'positive', 'negative', 'predicted_positive', 'predicted_negative', 
                #                     'group_label_positive', 'group_label_negative', 'true_negative', 'false_positive', 
                #                     'false_negative', 'true_positive', 'group_size', 'accuracy', 'precision_ppv', 'recall', 
                #                     'f1_score', 'group_prevalence', 'false_omission_rate', 'false_discovery_rate', 
                #                     'false_positive_rate', 'false_negative_rate', 'true_negative_rate', 'negative_predictive', 
                #                     'informedness', 'markedness', 'positive_likelihood_ratio', 'negative_likelihood_ratio', 
                #                     'prevalence_threshold', 'jaccard_index', 'fowlkes_mallows_index', 
                #                     'matthews_correlation_coefficient', 'diagnostic_odds_ratio', 'predicted_positive_rate_k', 
                #                     'predicted_positive_rate_g').collect()
                rows = out_task_1.select('total_records', 'race', 'attribute', 'for', 'fdr', 'fpr', 'fnr', 
                                         'pred_pos_ratio_g', 'pred_pos_ratio_k', 'group_size', 
                                         'fdr_disparity', 'fnr_disparity', 'for_disparity', 'fpr_disparity', 
                                         'pred_pos_ratio_k_disparity', 'pred_pos_ratio_g_disparity', 
                                         'fdr_parity', 'fnr_parity', 'for_parity', 'fpr_parity', 
                                         'pred_pos_ratio_k_parity', 'pred_pos_ratio_g_parity' 
                                        ).collect()

                content = SimpleTableReport(
                    'table table-striped table-bordered table-sm w-auto',
                    headers, rows)

                if display_text:
                    html = FairnessBiasReport({out},
                                '{sensitive}', baseline).generate()
                    visualization = {{
                        'job_id': '{job_id}', 'task_id': '{task_id}',
                        'title': '{title}',
                        'type': {{'id': 1, 'name': 'HTML'}},
                        'model': HtmlVisualizationModel(title='{title}'),
                        'data': json.dumps({{
                            'html': html,
                            'xhtml': '''
                                <a href="" target="_blank">
                                {title} ({open})
                                </a>'''
                        }}),
                    }}
                    emit_event(
                                'update task', status='COMPLETED',
                                identifier='{task_id}',
                                message=html,
                                type='HTML', title='{title}',
                                task={{'id': '{task_id}'}},
                                operation={{'id': {operation_id}}},
                                operation_id={operation_id})
                    {config}
                    # emit_event(
                    #             'update task', status='COMPLETED',
                    #             identifier='{task_id}',
                    #             message=base64.b64encode(fig_file.getvalue()),
                    #             type='IMAGE', title='{title}',
                    #             task={{'id': '{task_id}'}},
                    #             operation={{'id': {operation_id}}},
                    #             operation_id={operation_id})

                    # caipirinha_service.new_visualization(
                    #     config,
                    #     {user},
                    #     {workflow_id}, {job_id}, '{task_id}',
                    #     visualization, emit_event)
        """.format(sensitive=self.sensitive[0], label=self.label,
                   baseline=self.baseline, tau=self.tau,
                   score=self.score,
                   out=self.output, input=self.input,
                   display_text=display_text,
                   task_id=self.parameters['task_id'],
                   operation_id=self.parameters['operation_id'],
                   job_id=self.parameters['job_id'],
                   user=self.parameters['user'],
                   workflow_id=self.parameters['workflow_id'],
                   config=get_caipirinha_config(self.config, indentation=16),
                   title=gettext('Bias/Fairness Report'),
                   open=gettext('click to open'),
                   metric=self.type,
                   fairness_metric=self.fairness_metric,
                   metric_id=self.type,
                   column_name=self.column_name,
                   score_column_name=self.column_name.replace('parity',
                                                              'disparity'),
                   headers=[gettext('Group'), gettext('Acceptable'),
                            gettext('Value')]
                   ))
        return code
