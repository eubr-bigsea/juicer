import os
from gettext import gettext
from textwrap import dedent
import itertools
from juicer.operation import Operation


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """
    Converts a hex color code to RGBA format with specified transparency.

    Args:
    hex_color (str): Hex color code (e.g., #RRGGBB).
    alpha (float): Opacity value (0.0 - 1.0).

    Returns:
    str: RGBA color code (e.g., rgba(R, G, B, A)).
    """
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _hex_to_rgb(hexa):
    return "rgba({},{},{})".format(
        *tuple(int(hexa[i : i + 2], 16) for i in (1, 3, 5))
    )


def _rgb_to_rgba(rgb_value, alpha):
    return f"rgba{rgb_value[3:-1]}, {alpha})"


class VisualizationOperation(Operation):
    """
    Since 2.6
    """

    AGGR = {"COUNTD": "n_unique"}
    CHART_MAP_TYPES = ("scattermapbox", "densitymapbox")
    SCATTER_FAMILY = ("scatter", "indicator", "bubble")
    SUPPORTED_CHARTS = [
        "bar",
        "boxplot",
        "bubble",
        "donut",
        "funnel",
        "heatmap",
        "histogram2d",
        "histogram2dcontour",
        "horizontal-bar",
        "indicator",
        "line",
        "parcoords",
        "pie",
        "pointcloud",
        "scatter",
        "scattermapbox",
        "stacked-area",
        "stacked-area-100",
        "stacked-bar",
        "stacked-horizontal-bar",
        "sunburst",
        "treemap",
        "violin",
        'densitymapbox'
    ]

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

        self.type = self.get_required_parameter(parameters, "type")
        if self.type not in self.SUPPORTED_CHARTS:
            raise ValueError(gettext("Invalid chart type: {}".format(self.type)))
        self.display_legend = self.get_required_parameter(
            parameters, "display_legend"
        )

        if not self._is_map_family():
            self.x = self.get_required_parameter(parameters, "x")

            self.y_limit = None
            if len(self.x) > 1 or self.type in self.SCATTER_FAMILY:
                self.y_limit = 1

            self.y = self.get_required_parameter(parameters, "y")[: self.y_limit]

            self.aggregations = [y for y in self.y if y.get("aggregation")]
            self.literal = [y for y in self.y if not y.get("aggregation")]

            if self.aggregations and self.literal:
                raise(ValueError(gettext(
                    'Y-axis cannot include both values'
                     ' and aggregation functions')))

            self.x_axis = self.get_required_parameter(parameters, "x_axis")
            self.y_axis = self.get_required_parameter(parameters, "y_axis")

            for x in self.x:
                x["quantiles_list"] = []
                x["labels"] = []
                if x.get("binning") == "QUANTILES":
                    quantiles = [
                        int(q.strip())
                        for q in x["quantiles"].split(",")
                        if 0 <= int(q.strip()) <= 100
                    ]
                    tmp = [0] + quantiles + [100]
                    x["labels"] = [
                        f"{tmp[i]}-{tmp[i+1]}%" for i in range(len(tmp) - 1)
                    ]
                    x["quantiles_list"] = [0.01 * q for q in quantiles]

        self.animation = parameters.get("animation")
        self.auto_margin = parameters.get("auto_margin") in (True, 1, "1")
        self.blackWhite = parameters.get("blackWhite") in (True, 1, "1")
        self.bottom_margin = parameters.get("bottom_margin", 30)
        self.center_latitude = parameters.get("center_latitude")
        self.center_longitude = parameters.get("center_longitude")
        self.color_attribute = parameters.get("color_attribute")
        self.color_scale = parameters.get("color_scale", []) or []
        self.fill_opacity = parameters.get("fill_opacity")
        self.height = parameters.get("height")
        self.hole = parameters.get("hole", 30)
        self.latitude = parameters.get("latitude")
        self.left_margin = parameters.get("left_margin", 30)
        self.longitude = parameters.get("longitude")
        self.marker_size = parameters.get("marker_size")
        self.number_format = parameters.get("number_format")
        self.opacity = parameters.get("opacity")
        self.palette = parameters.get("palette")
        self.right_margin = parameters.get("right_margin", 30)
        self.size_attribute = parameters.get("size_attribute")
        self.smoothing = parameters.get("smoothing") in (1, True, "1")
        self.style = parameters.get("style")
        self.subgraph = parameters.get("subgraph")
        self.subgraph_orientation = parameters.get("subgraph_orientation")
        self.template_ = parameters.get("template", "none")
        self.text_attribute = parameters.get("text_attribute")
        self.text_info = parameters.get("text_info")
        self.text_position = parameters.get("text_position")
        self.title = parameters.get("title", "") or ""
        self.tooltip_info = parameters.get("tooltip_info")
        self.top_margin = parameters.get("top_margin", 30)
        self.width = parameters.get("width")
        self.zoom = parameters.get("zoom")
        self.max_width = parameters.get("max_width")
        self.max_height = parameters.get("max_height")

        self.hover_name = parameters.get('hover_name')
        self.hover_data = parameters.get('hover_data')
        self._compute_properties()

        #print('*' * 20)
        #print(self.hover_data)
        #print('*' * 20)

        self.has_code = len(named_inputs) == 1
        self.transpiler_utils.add_import(
            [
                "import json",
                "import numpy as np",
                "import itertools",
                "import plotly.express as px",
                "import plotly.graph_objects as go",
                "from plotly.subplots import make_subplots",
            ]
        )

        self.limit = parameters.get("limit")

        if self.blackWhite:
            self.transpiler_utils.add_import(
                "from plotly.colors import n_colors"
            )

        # if self.smoothing:
        #   self.transpiler_utils.add_import(
        #       'from scipy import signal as scipy_signal')

        self.task_id = parameters["task_id"]

        self.transparent_colors = []
        self.discrete_colors = []
        self._define_colors()
        if self.type not in self.CHART_MAP_TYPES:
            self.custom_colors = [
                (inx, y.get("color"))
                for inx, y in enumerate(self.y)
                if y.get("custom_color")
            ]
            self.shapes = [
                (inx, y.get("shape"))
                for inx, y in enumerate(self.y)
                if y.get("shape")
            ]
            # self.custom_shapes =
            # [(inx, y.get('shape')) for inx, y in enumerate(self.y)]

        self.supports_cache = False

    def _is_map_family(self):
        return self.type in self.CHART_MAP_TYPES

    def _compute_properties(self):
        """Compute properties used in template"""
        # Ranges
        lower_y = (
            self.y_axis.get("lowerBound", 'None') or 'None'
        )
        upper_y = self.y_axis.get("upperBound", 'None' or 'None')
        self.y_range = f"[{lower_y}, {upper_y}]"

        # Families
        self.pie_family = self.type in ('pie', 'donut')
        self.treemap_family = self.type in ('treemap', 'sunburst')
        self.stacked_family = self.type in ('stacked-bar',
                                            'stacked-horizontal-bar')
        self.horizontal_bar_family = self.type in ('horizontal-bar',
                                                    'stacked-horizontal-bar')
        self.filled_area_family = self.type in ('stacked-area',
                                                'stacked-area-100')
        self.scatter_family = self.type in ('scatter', 'bubble')

        self.use_color_scale = self.type in (
            'sunburst', 'treemap', 'histogram2dcontour', 'parcoords',
            'scattergeo', 'densitymapbox','histogram2d')
        self.use_color_discrete = not self.use_color_scale

    def _define_colors(self):
        if self.palette:
            if not self.pie_family:
                palette_options = itertools.cycle(self.palette)
                for y in self.y:
                    series_color = y.get("color") or next(palette_options)
                    if self.fill_opacity is not None and self.fill_opacity < 1:
                        series_color = _hex_to_rgba(
                            series_color, self.fill_opacity
                        )
                    self.discrete_colors.append(series_color)
            else:
                if self.fill_opacity < 1:
                    pass
                    self.discrete_colors = [
                        _hex_to_rgba(c, self.fill_opacity) for c in self.palette
                    ]
                else:
                    self.discrete_colors = self.palette
        else:
            pass

    def generate_code(self) -> str:
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "templates",
            "visualization_polars.tmpl",
        )
        with open(template_path) as f:
            self.template = f.read().strip()
        ctx = {
            "input": self.named_inputs["input data"],
            "out": self.named_outputs.get(
                "visualization", f"out_task_{self.order}"
            ),
            "op": self,
            "hex_to_rgba": _hex_to_rgba,
            "hex_to_rgb": _hex_to_rgb,
            "rgb_to_rgba": _rgb_to_rgba,
        }
        code = self.render_template(ctx, install_gettext=True)
        return dedent(code)
    def get_plotly_map_type(self):
        if self.type == 'densitymapbox':
            return 'density_mapbox'
        elif self.type == 'scattermapbox':
            return 'scatter_mapbox'
        else:
            raise ValueError('Not a map type')
