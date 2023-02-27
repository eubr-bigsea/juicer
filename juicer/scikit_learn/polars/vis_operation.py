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

        {%-if op.x|length > 1 or type == 'indicator' %}
        {%- set y_limit = 1 %}
        # More than 1 item for x-axis or type=indicator, limit y-axis to the 1st serie
        {%- else %}
        {%- set y_limit = None %}
        {%- endif %}
        aggregations = [
            {%- for y in op.y[:y_limit] %}
            {%- if y.attribute == '*' %}
            pl.count(){% if type == 'treemap' %}.cast(pl.Float64){% endif -%}
            .alias('aggr_{{loop.index0}}'),
            {%- else %}
            pl.{{op.AGGR.get(y.aggregation, y.aggregation.lower()) -}}
                ('{{y.attribute}}'){% if type == 'treemap' %}.cast(pl.Float64){% endif -%}
                .alias('aggr_{{loop.index0}}'),
            {%- endif %}
            {%- endfor %}
        ]

        {% if type != 'indicator' %}
        # Group data
        dimensions = [
            {%- for x in op.x %}pl.col('dim_{{loop.index0}}'), {%- endfor %}
        ]
        df = df.groupby(dimensions).agg(aggregations).sort(
            by=[{% if type != 'line' %}'aggr_0', {% endif %} 
            {%- for x in op.x %}'dim_{{loop.index0}}', {% endfor %}],
            descending=[{% if type != 'line' %}True, {% endif %}{% for x in op.x %}False, {% endfor %}])
        {%- else %}
        df = df.select(aggregations)
        {%- endif %}

        {%- if op.x|length > 1 %}
        # Fill the missing values. Otherwise, series may be wrongly 
        # sorted in the x-axis (if a new x-value is discovered after 
        # a previous serie is plot.
        # For example, for the first serie, there is no ('v1', 'a1') value,
        # but for there is a value for the second serie. The value 'a1'
        # would be displayed after values found in the first serie, 
        # causing x-axis to become wrongly sorted.
        tmp_df = (df.select('dim_0')
                    .unique()
                    .join(df.select('dim_1').unique(), how='cross')
                )
        df = df.join(tmp_df, on=['dim_0', 'dim_1'], how='outer').sort(
            ['dim_0', 'dim_1']).fill_null(0)
        {%- endif %}

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
                pandas_df,
                {%- if op.x|length == 1 or type == 'bar'%}
                x = 'dim_0',
                {%- else %}
                x=[{% for x in op.x %}'dim_{{loop.index0}}',{% endfor%}],
                {%- endif %}

                {%- if y_limit == 1%}
                y = 'aggr_0',
                {%- else %}
                y=[
                      {%- for y in op.y[:y_limit] -%}
                        'aggr_{{loop.index0}}',
                      {%- endfor -%}
                      ],
                {%- endif %}
                {%- if op.x|length > 1 and type in ('bar', ) -%}
                color='dim_1',
                {%- elif op.x|length > 1 and type not in ('scatter', ) -%}
                line_group='dim_1', color='dim_1',{% endif %}
                log_y={{op.y_axis.logScale}},
                color_discrete_sequence=colors,
                title='{{op.title}}',
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
            {%- if op.title %}
            title='{{op.title}}',
            {%- endif %}
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
            # color_discrete_sequence ={{op.palette}},
            title='{{op.title}}',
            labels=labels,
        )
        fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))

        {%- elif type in ('indicator', ) %}
        fig = go.Figure(go.Indicator(
            mode = "number",
            value = pandas_df.iloc[0][0],
            # title='{{op.title}}',
            # domain = {'x': [0, 1], 'y': [0, 1]}
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
        
        # Margins
        {%- if op.auto_margin %}
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(automargin=True)
        {%- else %}
        fig.update_layout(
            margin=dict(
                l={{op.left_margin}}, r={{op.right_margin}}, 
                t={{op.top_margin}}, b={{op.bottom_margin}})
        )
        {%- endif %}
        print(pandas_df)
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

        self.title = parameters.get('title', '') or ''
        self.hole = parameters.get('hole', 30)
        self.text_position= parameters.get('text_position')
        self.text_info = parameters.get('text_info')
        self.smoothing = parameters.get('smoothing') in (1, True, '1')
        self.palette = parameters.get('palette')
        self.color_scale = parameters.get('color_scale', []) or []
        self.auto_margin = parameters.get('auto_margin') in (True, 1, '1')
        self.right_margin = parameters.get('right_margin', 30)
        self.left_margin = parameters.get('left_margin', 30)
        self.top_margin = parameters.get('top_margin', 30)
        self.bottom_margin = parameters.get('bottom_margin', 30)

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
