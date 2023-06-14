import os
from gettext import gettext
from textwrap import dedent

from juicer.operation import Operation


def _hex_to_rgba(hexa, transparency):
    return 'rgba({},{},{},{})'.format(
        *tuple(int(hexa[i+1:i+3], 16) for i in (0, 2, 4)), transparency)


def _hex_to_rgb(hexa):
    return 'rgba({},{},{})'.format(
        *tuple(int(hexa[i+1:i+3], 16) for i in (0, 2, 4)))


def _rgb_to_rgba(rgb_value, alpha):
    return f"rgba{rgb_value[3:-1]}, {alpha})"


class VisualizationOperation(Operation):
    """
    Since 2.6
    """
    AGGR = {
        'COUNTD': 'n_unique'
    }
    CHART_MAP_TYPES = ('scattermapbox', )
    SCATTER_FAMILY = ('scatter', 'indicator', 'bubble')

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

        self.type = self.get_required_parameter(parameters, 'type')
        self.display_legend = self.get_required_parameter(
            parameters, 'display_legend')

        if self.type not in self.CHART_MAP_TYPES:
            self.x = self.get_required_parameter(parameters, 'x')

            self.y_limit = None
            if len(self.x) > 1 or self.type in SCATTER_FAMILY:
                self.y_limit = 1

            self.y = self.get_required_parameter(
                parameters, 'y')[:self.y_limit]

            self.aggregations = [y for y in self.y if y.get('aggregation')]
            self.literal = [y for y in self.y if not y.get('aggregation')]

            self.x_axis = self.get_required_parameter(parameters, 'x_axis')
            self.y_axis = self.get_required_parameter(parameters, 'y_axis')

            for x in self.x:
                x['quantiles_list'] = []
                x['labels'] = []
                if x.get('binning') == 'QUANTILES':
                    quantiles = [
                        int(q.strip()) for q in x['quantiles'].split(',')
                        if 0 <= int(q.strip()) <= 100]
                    tmp = [0] + quantiles + [100]
                    x['labels'] = [f"{tmp[i]}-{tmp[i+1]}%"
                                   for i in range(len(tmp) - 1)]
                    x['quantiles_list'] = [0.01 * q for q in quantiles]

        self.title = parameters.get('title', '') or ''
        self.hole = parameters.get('hole', 30)
        self.text_position = parameters.get('text_position')
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
        self.color_attribute = parameters.get('color_attribute')
        self.size_attribute = parameters.get('size_attribute')
        self.text_attribute = parameters.get('text_attribute')
        self.height = parameters.get('height')
        self.width = parameters.get('width')
        self.opacity = parameters.get('opacity')
        self.fill_opacity = parameters.get('fill_opacity')
        self.number_format = parameters.get('number_format')

        self.latitude = parameters.get('latitude')
        self.longitude = parameters.get('longitude')
        self.style = parameters.get('style')
        self.tooltip_info = parameters.get('tooltip_info')
        self.zoom = parameters.get('zoom')
        self.center_latitude = parameters.get('center_latitude')
        self.center_longitude = parameters.get('center_longitude')
        self.marker_size = parameters.get('marker_size')

        self.has_code = len(named_inputs) == 1
        self.transpiler_utils.add_import('import json')
        self.transpiler_utils.add_import('import numpy as np')
        self.transpiler_utils.add_import('import itertools')
        self.transpiler_utils.add_import('import plotly.express as px')
        self.transpiler_utils.add_import('import plotly.graph_objects as go')

        self.limit = parameters.get('limit')

        if self.blackWhite:
            self.transpiler_utils.add_import(
                'from plotly.colors import n_colors')

        # if self.smoothing:
        #   self.transpiler_utils.add_import(
        #       'from scipy import signal as scipy_signal')

        self.task_id = parameters['task_id']

        if self.type not in self.CHART_MAP_TYPES:
            self.custom_colors = [(inx, y.get('color')) for inx, y
                                  in enumerate(self.y) if y.get('custom_color')]
            self.shapes = [(inx, y.get('shape')) for inx, y
                           in enumerate(self.y) if y.get('shape')]
            # self.custom_shapes =
            # [(inx, y.get('shape')) for inx, y in enumerate(self.y)]

        self.supports_cache = False

    def generate_code(self) -> str:
        template_path = os.path.join(
            os.path.dirname(__file__), '..', 'templates',
            'visualization_polars.tmpl')
        with open(template_path) as f:
            self.template = f.read().strip()
        ctx = {
            'input': self.named_inputs['input data'],
            'out': self.named_outputs.get('visualization',
                                          f'out_task_{self.order}'),
            'op': self,
            'hex_to_rgba': _hex_to_rgba,
            'hex_to_rgb': _hex_to_rgb,
            'rgb_to_rgba': _rgb_to_rgba,
        }
        code = self.render_template(ctx)
        return dedent(code)
