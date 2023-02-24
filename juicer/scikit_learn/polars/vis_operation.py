from gettext import gettext
from textwrap import dedent

from juicer.operation import Operation


class VisualizationOperation(Operation):
    """
    Since 2.6
    """
    AGGR = {
        'COUNTD': 'n_unique'
    }
    template = """
        df = {{input}}
        # Prepare X-Axis values

        {%- for x in op.x %}
        {%- if x.binning == 'EQUAL_INTERVAL' %}
        # Binning with equal interval
        col = pl.col('{{x.attribute}}')
        df = df.with_columns([
            (col - col.min()).alias('_diff'),
            ((col.max() - col.min())
                    / {{x.bins}}).alias('_bin_size')
        ])
        df = df.with_columns([
            pl.min([pl.col('_diff') // pl.col('_bin_size'), 
                {{x.bins - 1}}]).cast(pl.Int16).alias('dim_{{loop.index0}}')
        ])
        {%- elif x.binning == 'EQUAL_INTERVAL' %}
        {%- endif %}
        {%- endfor %}

        # Group data
        df = df.groupby([
            pl.col({%- for x in op.x %}'dim_{{loop.index0}}'{%- endfor %})
        ]).agg([
            {%- for y in op.y %}
            {%- if y.attribute == '*' %}
            pl.count().alias('aggr_{{loop.index0}}'),
            {%- else %}
            pl.{{op.AGGR.get(y.aggregation, y.aggregation.lower()) -}}
                ('{{y.attribute}}').alias('aggr_{{loop.index0}}'),
            {%- endif %}
            {%- endfor %}
        ]).sort('dim_0').collect().to_pandas()
        labels = {
            {%- for y in op.y %}
            'aggr_{{loop.index0}}': 
            {%- if y.label -%}
            '{{y.label}}',
            {%- else -%}
            '{{op.AGGR.get(y.aggregation, y.aggregation.lower()) -}}
                ({{y.attribute}})',
            {%- endif %}
            {%- endfor -%} 
        }
        colors = {{op.palette}}
        {%- if op.type.lower() == 'line' %}
        fig = px.line(
            df, x='dim_0', y=[
                      {%- for y in op.y -%}
                        'aggr_{{loop.index0}}',
                      {%- endfor -%}
                      ],
            log_y={{op.y_axis.logScale}},
            color_discrete_sequence=colors,
            title='{{op.parameters.workflow.name}}',
        )
        
        fig.for_each_trace(lambda t: t.update(name = labels[t.name]))
        fig.update_layout(
             xaxis_title='{{op.x_axis.label}}',
             #yaxis_title='{{op.y_axis.label}}'
        )
        fig.update_yaxes(
            title='{{op.y_axis.label}}', showgrid=True, 
            gridcolor="rgba(255,0,0,.10)", 
            visible={{op.y_axis.display}}, tickformat='.3f',
                tickprefix='{{op.y_axis.prefix}}', 
                ticksuffix='{{op.y_axis.suffix}}', 
                #showtickprefix="first", 
                #showticksuffix="last"
        )
        fig.update_layout(legend=dict(
            title='Legend',
            orientation="v",
            yanchor="auto",
            y=-.1,
            xanchor="right",  # changed
            x=-0.2,
            ),
            width=1200,
            autosize=True,
        )
        sizes = [{% for y in op.y -%}
            {%- if y.strokeSize %}
            {{y.strokeSize}},
            {%- else %}
            1,
            {%- endif %}
            {%- endfor %}]

        styles = [{% for y in op.y -%}
            {%- if y.stroke %}
            '{{y.stroke}}',
            {%- else %}
            'solid',
            {%- endif %}
            {%- endfor %}]
        for s, z, style in zip(fig.data, sizes, styles):
            s.line["dash"] = style
            #s.marker.symbol = next(markers)
            s.marker.size = z

        d = json.loads(fig.to_json())
        del d.get('layout')['template']
        print(d)

        emit_event(
            'update task', status='COMPLETED',
            identifier='{{op.task_id}}',
            message=d,
            type='PLOTLY', title='',
            task={'id': '{{op.task_id}}'},
        )
        
        {%- endif %}

        {{out}} = None
    """
        
    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

        self.type = self.get_required_parameter(parameters, 'type')
        self.display_legend = self.get_required_parameter(
            parameters, 'display_legend')
        self.palette = parameters.get('palette')
        self.x = self.get_required_parameter(parameters, 'x')
        self.y = self.get_required_parameter(parameters, 'y')
        self.x_axis = self.get_required_parameter(parameters, 'x_axis')
        self.y_axis = self.get_required_parameter(parameters, 'y_axis')

        self.has_code = len(named_inputs) == 1
        self.transpiler_utils.add_import('import json')
        self.transpiler_utils.add_import('import numpy as np')
        self.transpiler_utils.add_import('import plotly.express as px')
        self.task_id = parameters['task_id']

        self.supports_cache = False

    def generate_code(self):
        ctx = {
            'input': self.named_inputs['input data'],
            'out': self.named_outputs.get('visualization', f'out_task_{self.order}'),
            'op': self,
        }
        code = self.render_template(ctx)
        return dedent(code)
