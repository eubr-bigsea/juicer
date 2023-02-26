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
        {%- set type = op.type.lower() %}
        df = {{input}}
        # Prepare X-Axis values

        {%- for x in op.x %}
        col = pl.col('{{x.attribute}}')
        {%- if x.binning == 'EQUAL_INTERVAL' %}
        # Binning with equal interval
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
        {%- else %}
        df = df.with_columns([col.alias('dim_{{loop.index0}}')])
        {%- endif %}
        {%- endfor %}

        colors = {{op.palette}}

        {%-if op.x|length > 1%}
        {%- set y_limit = 1 %}
        # More than 1 item for x-axis, limit y-axis to the 1st serie
        {%- else %}
        {%- set y_limit = None %}
        {%- endif %}
        aggregations = [
            {%- for y in op.y[:y_limit] %}
            {%- if y.attribute == '*' %}
            pl.count().alias('aggr_{{loop.index0}}'),
            {%- else %}
            pl.{{op.AGGR.get(y.aggregation, y.aggregation.lower()) -}}
                ('{{y.attribute}}').alias('aggr_{{loop.index0}}'),
            {%- endif %}
            {%- endfor %}
        ]

        # Group data
        df = df.groupby([
            {%- for x in op.x %}pl.col('dim_{{loop.index0}}'), {%- endfor %}
        ]).agg(aggregations).sort(
            by=['aggr_0', {% for x in op.x %}'dim_{{loop.index0}}', {% endfor %}],
            descending=[True, {% for x in op.x %}False, {% endfor %}])

        pandas_df = df.collect().to_pandas()
        labels = {
            'variable': 'Série',
            {%- for y in op.y[:y_limit] %}
            'aggr_{{loop.index0}}':
            {%- if y.label -%}
            '{{y.label}}',
            {%- else -%}
            '{{op.AGGR.get(y.aggregation, y.aggregation.lower()) -}}
                ({{y.attribute}})',
            {%- endif %}
            {%- endfor -%}
            {%- for x in op.x -%}
                'dim_{{loop.index0}}': '{{x.displayLabel or x.attribute}}', {%- endfor %}
        }

        #  Chart definition
        {%- if type in ('line', 'filled-area', 'stacked-filled-area',
            'bar', 'stacked-bar', 'horizontal-bar',
            'stacked-horizontal-bar', 'scatter') %}
        fig = px.
            {%- if type in ('stacked-filled-area', 'filled-area') -%}
            area
            {%- elif type in ('stacked-bar', 'stacked-horizontal-bar', 'horizontal-bar') -%}
            bar
            {%- else %}{{type}}{% endif -%}(
                pandas_df, x='dim_0', y=[
                      {%- for y in op.y[:y_limit] -%}
                        'aggr_{{loop.index0}}',
                      {%- endfor -%}
                      ],
                {%- if op.x|length > 1 and type not in ('scatter', 'bar') -%}
                line_group='dim_1', color='dim_1',{% endif %}
                log_y={{op.y_axis.logScale}},
                color_discrete_sequence=colors,
                title='{{op.parameters.workflow.name}}',
                {%- if type in ('bar', 'horizontal-bar') %}barmode='group',{% endif %}
                {%- if type in ('stacked-bar', 'stacked-horizontal-bar') %}barmode='stack',{% endif %}
                {%- if type in ('stacked-horizontal-bar', 'horizontal-bar') %}orientation='h',{% endif %}
                {%- if type in ('filled-area', 'stacked-filled-area') %}orientation='v',{% endif %}
                {%- if type in ('stacked-filled-area', ) %}groupnorm='percent',{% endif %}
                labels=labels,
                {%- if type == 'line' and op.smoothing %}
                # https://github.com/plotly/plotly.py/issues/2812
                render_mode='svg',
                {%- endif %}
        )

        fig.for_each_trace(lambda t: t.update(name = labels.get(t.name, t.name)))
        fig.update_layout(
             xaxis_title='{{op.x_axis.label}}',
        )
        fig.update_yaxes(
            title='{{op.y_axis.label}}', showgrid=True,
            gridcolor="rgba(255,0,0,.10)",
            visible={{op.y_axis.display}}, tickformat=
                {%- if type == 'stacked-filled-area' %}None{% else %}'.3f'{% endif %},
                tickprefix='{{op.y_axis.prefix}}',
                ticksuffix=
                {%- if type == 'stacked-filled-area' %}'%'{% else %}'{{op.y_axis.suffix}}'{% endif %},
                #showtickprefix="first",
                #showticksuffix="last"
        )

        {%- if op.smoothing %}
        fig.update_traces(line={'shape': 'spline', 'smoothing': 0.4})
        {%- endif %}

        {%- if type == 'line' %}
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
        {%- endif %}

        {%- elif type in ('pie', 'donut') %}
        fig = px.pie(pandas_df, values='aggr_0',
            names='dim_0',
            color_discrete_sequence=colors,
            title='{{op.parameters.workflow.name}}',
            {%- if type == 'donut' %}hole={{op.hole * 0.01}},{% endif %}
            #width=800, height=400,
            #{{op.hole}},
            labels=labels,
        )
        {%-if op.text_position or op.text_info %}
        fig.update_traces(
            pull=[0.01] * {{op.y |length}},
            {%- if op.text_position %}textposition='{{op.text_position}}',
            {%- endif %}
            {%- if op.text_info != '' %}textinfo='{{op.text_info}}',{% endif %}
        )
        {%- endif %}
        
        {%- elif type in ('treemap',) %}
        fig = px.treemap(
            pandas_df, 
            path=[{% for x in op.x %}'dim_{{loop.index0}}', {% endfor %}],
            values='aggr_0',
            color='aggr_0',
            color_continuous_scale={{op.color_scale}},
            color_discrete_sequence ={{op.palette}},
            title='{{op.parameters.workflow.name}}',
            labels=labels,
        )
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))

        {%- elif type in ('indicator', ) %}
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 450,
            title = {'text': "Speed"},
            domain = {'x': [0, 1], 'y': [0, 1]}
        ))
        {%- endif %}

        {%-if op.display_legend == 'HIDE' %}
        fig.update_layout(showlegend=False)
        {%- elif op.display_legend != 'AUTO' %}
        # Legend
        fig.update_layout(
            showlegend=True,
            legend=dict(
                title='Legend',
                orientation="{%if 'CENTER' in op.display_legend %}h{%else%}v{%endif%}",
                yanchor="{% if 'BOTTOM' in op.display_legend %}bottom
                {%- else %}top{% endif %}",
                y={% if 'BOTTOM' in op.display_legend %}-.30
                {%- else %}.99{% endif %},
                xanchor="{{op.display_legend.lower().replace('bottom_', '')}}",
                x={% if 'CENTER' in op.display_legend %}0.5
                {%- elif 'LEFT' in op.display_legend %}.1
                {%- else %}.99{% endif %}
            )
        )
        {%- endif %}
        d = json.loads(fig.to_json())
        del d.get('layout')['template']
        emit_event(
            'update task', status='COMPLETED',
            identifier='{{op.task_id}}',
            message=d,
            type='PLOTLY', title='',
            task={'id': '{{op.task_id}}'},
        )

        {{out}} = None
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

        self.type = self.get_required_parameter(parameters, 'type')
        self.display_legend = self.get_required_parameter(
            parameters, 'display_legend')
        self.x = self.get_required_parameter(parameters, 'x')
        self.y = self.get_required_parameter(parameters, 'y')
        self.x_axis = self.get_required_parameter(parameters, 'x_axis')
        self.y_axis = self.get_required_parameter(parameters, 'y_axis')

        self.hole = parameters.get('hole', 30)
        self.text_position= parameters.get('text_position')
        self.text_info = parameters.get('text_info')
        self.smoothing = parameters.get('smoothing') in (1, True, '1')
        self.palette = parameters.get('palette')
        self.color_scale = parameters.get('color_scale', []) or []

        self.has_code = len(named_inputs) == 1
        self.transpiler_utils.add_import('import json')
        self.transpiler_utils.add_import('import numpy as np')
        self.transpiler_utils.add_import('import plotly.express as px')
        self.transpiler_utils.add_import('import plotly.graph_objects as go')
        #if self.smoothing:
        #   self.transpiler_utils.add_import(
        #       'from scipy import signal as scipy_signal')
            
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
