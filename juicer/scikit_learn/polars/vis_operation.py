import os
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
    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

        self.type = self.get_required_parameter(parameters, 'type')
        self.display_legend = self.get_required_parameter(
            parameters, 'display_legend')
        self.x = self.get_required_parameter(parameters, 'x')

        self.y_limit = None
        if len(self.x) > 1 or type == 'indicator':
            self.y_limit = 1

        self.y = self.get_required_parameter(parameters, 'y')[:self.y_limit]

        self.aggregations = [y for y in self.y if y.get('aggregation')]
        self.literal = [y for y in self.y if not y.get('aggregation')]

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
        self.template_ = parameters.get('template', 'none')
        self.blackWhite = parameters.get('blackWhite') in (True, 1, '1')
        self.subgraph = parameters.get('subgraph')
        self.subgraph_orientation = parameters.get('subgraph_orientation')
        self.animation = parameters.get('animation')
        self.height = parameters.get('height')
        self.width = parameters.get('width')
        self.opacity = parameters.get('opacity')

        self.has_code = len(named_inputs) == 1
        self.transpiler_utils.add_import('import json')
        self.transpiler_utils.add_import('import numpy as np')
        self.transpiler_utils.add_import('import plotly.express as px')
        self.transpiler_utils.add_import('import plotly.graph_objects as go')

        if self.blackWhite:
            self.transpiler_utils.add_import('from plotly.colors import n_colors')
            
        if self.type == 'line':
            self.transpiler_utils.add_import('from itertools import zip_longest')
        #if self.smoothing:
        #   self.transpiler_utils.add_import(
        #       'from scipy import signal as scipy_signal')
            
        self.task_id = parameters['task_id']

        self.supports_cache = False

    def generate_code(self):
        template_path = os.path.join(
            os.path.dirname(__file__), '..', 'templates', 
            'visualization_polars.tmpl')
        with open(template_path) as f:
            self.template = f.read().strip()
        ctx = {
            'input': self.named_inputs['input data'],
            'out': self.named_outputs.get('visualization', f'out_task_{self.order}'),
            'op': self,
        }
        code = self.render_template(ctx)
        return dedent(code)
