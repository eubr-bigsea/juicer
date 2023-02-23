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
                {{x.bins - 1}}]).cast(pl.Int16).alias('{{x.attribute}}_dim')
        ])
        {%- elif x.binning == 'EQUAL_INTERVAL' %}
        {%- endif %}
        {%- endfor %}

        # Group data
        df = df.groupby([
            pl.col({%- for x in op.x %}'{{x.attribute}}_dim'{%- endfor %})
        ]).agg([
            {%- for y in op.y %}
            {%- if y.attribute == '*' %}
            pl.count(),
            {%- else %}
            pl.{{op.AGGR.get(y.aggregation, y.aggregation.lower()) -}}
                ('{{y.attribute}}'),
            {%- endif %}
            {%- endfor %}
        ])
        import pdb; pdb.set_trace()
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
        self.transpiler_utils.add_import('import numpy as np')

    def generate_code(self):
        ctx = {
            'input': self.named_inputs['input data'],
            'out': self.named_outputs.get('visualization', f'out_task_{self.order}'),
            'op': self,
        }
        code = self.render_template(ctx)
        return dedent(code)
