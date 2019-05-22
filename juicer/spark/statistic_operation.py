# coding=utf-8

from textwrap import dedent

from juicer.operation import Operation
from juicer.spark.vis_operation import get_caipirinha_config


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

        self.legend = parameters.get(self.LEGEND_PARAM, [])
        self.title = parameters.get(self.TITLE_PARAM,
                                    _('Kaplar-Meier survival'))
        self.supports_cache = False
        self.has_code = all([
            len(self.named_inputs) == 1,
            any([len(self.named_outputs) == 1, self.contains_results()])])

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        display_image = self.parameters['task']['forms'].get(
            'display_image', {'value': 1}).get('value', 1) in (1, '1')
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
            display_image = {display_image}
            if pivot:
                groups = df.groupby([pivot])
                {out} = None

                if display_image:
                    fig = plt.figure()
                    plt.title('{title}')
                    ax1 = fig.add_subplot(111)
                    ax1.set_xlabel('{t}')
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
                    if display_image:
                        kmf.survival_function_.plot(ax=ax1)
                if display_image:
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
            if display_image:
                fig_file = BytesIO()

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
                emit_event(
                            'update task', status='COMPLETED',
                            identifier='{task_id}',
                            message=base64.b64encode(fig_file.getvalue()),
                            type='IMAGE', title='{title}',
                            task={{'id': '{task_id}'}},
                            operation={{'id': {operation_id}}},
                            operation_id={operation_id})
                            
                # caipirinha_service.new_visualization(
                #     config,
                #     {user},
                #     {workflow_id}, {job_id}, '{task_id}',
                #     visualization, emit_event)
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
                   display_image=display_image
                   ))]
        return ''.join(code)


class CoxProportionalHazardsOperation(Operation):
    ATTRIBUTES_PARAM = 'attributes'
    Y_ATTRIBUTE_PARAM = 'y_attribute'
    TIME_ATTRIBUTE_PARAM = 'time_attribute'

    LEGEND_PARAM = 'legend'
    TITLE_PARAM = 'title'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        required = [self.Y_ATTRIBUTE_PARAM, self.TIME_ATTRIBUTE_PARAM]

        self.time_attribute = None
        self.y_attribute = None
        for r in required:
            if r in parameters and len(parameters[r]) > 0:
                setattr(self, r, parameters.get(r)[0])
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                        r, self.__class__))
        self.attributes = parameters.get('attributes', [])

        self.input = self.named_inputs.get('input data')
        self.output = self.named_outputs.get('output data',
                                             'out_task_{}'.format(self.order))
        self.has_code = len(self.named_inputs) == 1

        self.legend = parameters.get(self.LEGEND_PARAM, [])
        self.title = parameters.get(self.TITLE_PARAM,
                                    _('Cox Proportional Hazard'))
        self.supports_cache = False

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):
        display_image = self.parameters['task']['forms'].get(
            'display_image', {'value': 1}).get('value', 1) not in (0, '0')
        code = [
            dedent("""
            from io import BytesIO
            from juicer.lib import cox
            from juicer.service import caipirinha_service
            from juicer.spark.vis_operation import HtmlVisualizationModel
            from matplotlib.ticker import FormatStrFormatter
            import matplotlib.pyplot as plt
            import base64
            """),
            dedent(get_caipirinha_config(self.config)),
            dedent("""
            df = {input}.toPandas()
            y_name = '{y_name}'
            t_name = '{t_name}'
            columns = {attributes}
            result = cox(df, y_name, t_name, columns)
            final_names = [x[0] for x in result.columns.values]
            {out} = spark_session.createDataFrame(result, final_names)
            labels = list(result.keys().levels[0])

            display_image = {display_image}
            if display_image:
                rows = len(labels) - 1
                fig, axes = plt.subplots(ncols=1, nrows=rows, figsize=(8, 8))
                # axes.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                # axes.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

                plt.title('{title}')
                for i, l in enumerate(labels):
                    if l != 'Time':
                        plt.subplot(rows, 1, i + 1)
                        plt.scatter(result["Time"], result[l])

                fig.subplots_adjust()
                fig_file = BytesIO()
                # fig.tight_layout(rect=[0, 0.03, 1, 0.95])

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
                emit_event(
                            'update task', status='COMPLETED',
                            identifier='{task_id}',
                            message=base64.b64encode(fig_file.getvalue()),
                            type='IMAGE', title='{title}',
                            task={{'id': '{task_id}'}},
                            operation={{'id': {operation_id}}},
                            operation_id={operation_id})

                # caipirinha_service.new_visualization(
                #     config,
                #     {user},
                #     {workflow_id}, {job_id}, '{task_id}',
                #     visualization, emit_event)


            """.format(input=self.input, y_name=self.y_attribute,
                       t_name=self.time_attribute,
                       out=self.output,
                       title=self.title,
                       task_id=self.parameters['task_id'],
                       operation_id=self.parameters['operation_id'],
                       job_id=self.parameters['job_id'],
                       user=self.parameters['user'],
                       workflow_id=self.parameters['workflow_id'],
                       display_image=display_image,
                       attributes=repr(self.attributes)))
        ]
        return ''.join(code)
