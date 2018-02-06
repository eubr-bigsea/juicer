import json
from textwrap import dedent

from juicer.operation import Operation
from juicer.spark.vis_operation import get_caipirinha_config


class PearsonCorrelation(Operation):
    """
    Calculates the correlation of two columns of a DataFrame as a double value.
    @deprecated: It should be used as a function in expressions
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.has_code = len(self.inputs) == 1

    def generate_code(self):
        output = self.outputs[0] if len(self.outputs) else '{}_tmp'.format(
            self.inputs[0])
        code = """{} = {}.corr('{}', '{}')""".format(
            output, self.inputs[0], self.attributes[0], self.attributes[1])

        return dedent(code)


class KaplanMeierSurvivalOperation(Operation):
    PIVOT_ATTRIBUTE_PARAM = 'pivot_attribute'
    DURATION_ATTRIBUTE_PARAM = 'duration_attribute'
    EVENT_OBSERVED_ATTRIBUTE = 'event_observed_attribute'
    LEGEND_PARAM = 'legend'
    TITLE_PARAM = 'title'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        required = [self.DURATION_ATTRIBUTE_PARAM,
                    self.EVENT_OBSERVED_ATTRIBUTE]

        self.pivot_attribute = (parameters.get(
            self.PIVOT_ATTRIBUTE_PARAM) or [None])[0]
        self.duration_attribute = None
        self.event_observed_attribute = None
        for r in required:
            if r in parameters and len(parameters[r]) > 0:
                setattr(self, r, parameters.get(r)[0])
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        r, self.__class__))
        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.input = self.named_inputs.get('input data')

        self.has_code = any([len(named_outputs) > 0 and
                             len(self.named_inputs) == 1,
                             self.contains_results()])
        self.legend = parameters.get(self.LEGEND_PARAM, [])
        self.title = parameters.get(self.TITLE_PARAM,
                                    _('Kaplar-Meier survival'))
        self.supports_cache = False

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        code = [
            dedent("""
            import matplotlib.pyplot as plt
            import pandas as pd
            import base64

            from io import BytesIO

            from juicer.service import caipirinha_service
            from juicer.spark.vis_operation import HtmlVisualizationModel
            from lifelines import KaplanMeierFitter"""),
            dedent(get_caipirinha_config(self.config)),
            dedent("""
            kmf = KaplanMeierFitter()
            df = {input}.toPandas()
            pivot = '{pivot}'
            labels = {legend}
            fig = plt.figure()
            plt.title('{title}')
            ax1 = fig.add_subplot(111)
            ax1.set_xlabel('{t}')
            fig_file = BytesIO()
            if pivot:
                groups = df.groupby([pivot])
                {out} = None


                for i, group_id in enumerate(groups.groups):
                    if len(labels) > i:
                        label = labels[i]
                    else:
                        label = "{{}}={{}}".format(pivot, group_id)

                    data = groups.get_group(group_id)
                    kmf.fit(data['{duration}'],
                        event_observed=data['{event_observed}'],
                        label=label)

                    pd_df = kmf.survival_function_
                    pd_df = pd.concat([
                        pd_df,
                        pd.Series([group_id] * len(pd_df)),
                        pd.Series(range(len(pd_df)))], axis=1)

                    if {out} is None:
                        {out} = spark_session.createDataFrame(
                            pd_df, ['{f}', '{v}', '{t}'])
                    else:
                        {out} = {out}.union(spark_session.createDataFrame(
                            pd_df, ['{f}', '{v}', '{t}']))
                    kmf.survival_function_.plot(ax=ax1)

                ax1.set_ylabel('')
            else:
                kmf.fit(df['{duration}'], event_observed=df['{event_observed}'],
                    label=labels[0] if labels else None)
                pd_df = kmf.survival_function_
                pd_df = pd.concat([
                    pd_df,
                    pd.Series(range(len(pd_df)))], axis=1)
                {out} = spark_session.createDataFrame(
                            pd_df, ['{f}', '{t}'])
                kmf.survival_function_.plot(ax=ax1)

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(fig_file, format='png')
            visualization = {{
                'job_id': '{job_id}',
                'task_id': '{task_id}',
                'title': '{title}',
                'type': {{
                    'id': 1,
                    'name': 'HTML'
                }},
                'model': HtmlVisualizationModel(title='{title}'),
                'data': json.dumps({{
                    'html': '<img src="data:image/png;base64, ' +
                        base64.b64encode(fig_file.getvalue()) + '"/>'
                }}),
            }}

            caipirinha_service.new_visualization(
                config,
                {user},
                {workflow_id}, {job_id}, '{task_id}',
                visualization, emit_event)
        """.format(out=self.output,
                   input=self.input,
                   pivot=self.pivot_attribute or '',
                   duration=self.duration_attribute,
                   event_observed=self.event_observed_attribute,
                   title=self.title,
                   legend=repr(self.legend),
                   f=_('KM_estimate'),
                   v=_('value'),
                   t=_('timeline'),
                   task_id=self.parameters['task_id'],
                   operation_id=self.parameters['operation_id'],
                   job_id=self.parameters['job_id'],
                   user=self.parameters['user'],
                   workflow_id=self.parameters['workflow_id'],
                   ))]
        return ''.join(code)
