import pytest

from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from tests.scikit_learn import util
# Common tests



legend_params_and_expected = [
    ({"display_legend": "AUTO"}, {"o": None, "pos": [None, None], "show": None}),
    (
        {"display_legend": "HIDE"},
        {"o": None, "pos": [None, None], "show": False},
    ),
    (
        {"display_legend": "LEFT"},
        {"o": "v", "pos": ["left", "top"], "show": True},
    ),
    (
        {"display_legend": "RIGHT"},
        {"o": "v", "pos": ["right", "top"], "show": True},
    ),
    (
        {"display_legend": "CENTER"},
        {"o": "h", "pos": ["center", "top"], "show": True},
    ),
    (
        {"display_legend": "BOTTOM_LEFT"},
        {"o": "v", "pos": ["left", "bottom"], "show": True},
    ),
    (
        {"display_legend": "BOTTOM_RIGHT"},
        {"o": "v", "pos": ["right", "bottom"], "show": True},
    ),
    (
        {"display_legend": "BOTTOM_CENTER"},
        {"o": "h", "pos": ["center", "bottom"], "show": True},
    ),
    ({}, {"o": None, "pos": [None, None], "show": False}),
]


@pytest.fixture(scope="session")
def get_df():
    return util.titanic_polars()

@pytest.fixture
def get_arguments(updated_params):
    result = {
        "parameters": {
            "type": "sunburst",
            "display_legend": "HIDE",
            "x": [
                {
                    "bins": 20,
                    "binSize": 10,
                    "emptyBins": "ZEROS",
                    "decimal_places": 2,
                    "group_others": True,
                    "sorting": "NATURAL",
                    "attribute": "pclass",
                }
            ],
            "color_scale": [
                "#000000",
                "#e60000",
                "#e6d200",
                "#ffffff",
                "#a0c8ff",
            ],
            "y": [
                {
                    "attribute": "*",
                    "aggregation": "COUNT",
                    "displayOn": "left",
                    "decimal_places": 2,
                    "strokeSize": 0,
                    "enabled": True,
                }
            ],
            "x_axis": {
                "logScale": False,
                "display": True,
                "displayLabel": True,
                "decimal_places": 2,
            },
            "y_axis": {
                "logScale": False,
                "display": True,
                "displayLabel": True,
                "label": None,
                "multiplier": None,
                "decimal_places": 2,
            },
            "task_id": "0",
        },
        "named_inputs": {
            "input data": "iris",
        },
        "named_outputs": {"output data": "out"},
    }
    result["parameters"].update(updated_params)
    return result


@pytest.fixture
def generated_chart(get_arguments, get_df):
    instance = VisualizationOperation(**get_arguments)
    vis_globals = dict(iris=get_df, emit_event=util.emit_event)
    code = "\n".join(
        [
            "import plotly.graph_objects as go",
            "import plotly.express as px",
            "import json",
            instance.generate_code(),
        ]
    )
    result = util.execute(code, vis_globals)
    # breakpoint()
    generated_chart_code = result.get("d")
    data = generated_chart_code["data"]
    layout = generated_chart_code["layout"]

    return data, layout


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        ({"type": "sunburst"}, "sunburst"),
        # ({"type": "heatmap"}, 'heatmap'),
        ({"type": "treemap"}, "treemap"),
    ],
)
def test_data_type_success(generated_chart, expected):
    data, _ = generated_chart

    chart_type = data[0].get("type")
    assert chart_type is not None, "Field 'type' not found in the data object"
    assert chart_type == expected, "Incorrect value for the 'type' field"


def xtest_data_type_fail(get_arguments):
    data, _ = generated_chart(get_arguments)
    sunburst_data = data[0]
    chart_type = sunburst_data.get("type")
    assert chart_type is not None, "Field 'type' not found in the data object"
    assert chart_type == "sunburst", "Incorrect value for the 'type' field"


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        ({"title": "First test"}, "First test"),
        ({"title": "Acentuação"}, "Acentuação"),
        ({"title": "sPace$ & $1mb%o&s"}, "sPace$ & $1mb%o&s"),
        ({}, None),
    ],
)
def test_title_success(generated_chart, expected):
    _, layout = generated_chart
    title = layout.get("title", {}).get("text")
    assert (
        title is not None or expected is None
    ), "Field 'title' not found in the data object"
    assert title == expected, "Incorrect value for the 'title' field"


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        ({"number_format": "1,234.56"}, "1,234.56"),
        ({"number_format": "1,234.56"}, "1,234.56"),
        ({"number_format": "1.234,56"}, "1.234,56"),
        ({"number_format": "1 234.56"}, "1 234.56"),
        ({"number_format": "1 234,56"}, "1 234,56"),
        ({}, None),
    ],
)
def test_layout_number_format_success(generated_chart, expected):
    _, layout = generated_chart
    separators = layout.get("separators")
    assert (
        separators == expected
    ), "Incorrect values for the 'width' or 'height' field"


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        ({"height": 200, "width": 300}, (300, 200)),
        ({}, (None, None)),
    ],
)
def test_layout_size_success(generated_chart, expected):
    _, layout = generated_chart
    (width, height) = layout.get("width"), layout.get("height")
    assert (
        width,
        height,
    ) == expected, "Incorrect values for the 'width' or 'height' field"


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        ({"auto_margin": False}, {"t": 30, "l": 30, "r": 30, "b": 30}),
        ({}, {"t": 30, "l": 30, "r": 30, "b": 30}),
    ],
)
def test_layout_auto_margin_false_success(generated_chart, expected):
    _, layout = generated_chart
    layout_margin = layout.get("margin")
    assert (
        layout_margin is not None
    ), "Field 'margin' not found in the layout object"
    assert layout_margin == expected, "Incorrect values for the 'margin' field"


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        ({"auto_margin": True}, {}),
    ],
)
def test_layout_auto_margin_true_success(generated_chart, expected):
    _, layout = generated_chart
    auto_margin_x, auto_margin_y = (layout.get("xaxis").get('automargin'),
                                    layout.get("yaxis").get('automargin'))
    assert auto_margin_x, "Incorrect values for the 'margin' field"
    assert auto_margin_y, "Incorrect values for the 'margin' field"


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        (
            {
                "left_margin": 11,
                "right_margin": 13,
                "top_margin": 17,
                "bottom_margin": 23,
            },
            {"b": 23, "t": 17, "l": 11, "r": 13},
        ),
        (
            {
                "left_margin": 20,
                "right_margin": 20,
                "top_margin": 35,
                "bottom_margin": 45,
            },
            {"b": 45, "t": 35, "l": 20, "r": 20},
        ),
    ],
)
def test_layout_margin_success(generated_chart, expected):
    _, layout = generated_chart
    layout_margin = layout.get("margin")
    assert (
        layout_margin is not None
    ), "Field 'margin' not found in the layout object"
    assert layout_margin == expected, "Incorrect values for the 'margin' field"



@pytest.mark.parametrize("updated_params, expected", [
    ({}, {})
])
def test_data_hovertemplate(generated_chart, expected):
    data, _ = generated_chart
    sunburst_data = data[0]
    hovertemplate = sunburst_data.get("hovertemplate")
    assert True or expected
    assert (
        hovertemplate is not None
    ), "Field 'hovertemplate' not found in data object"


@pytest.mark.parametrize("updated_params, expected", legend_params_and_expected)
def test_layout_showlegend_success(generated_chart, expected):
    """Notice that generated_chart has a parameter named updated_params. The
    value for updated_params is automatically passed to the fixture if there is
    a name match.
    See https://stackoverflow.com/a/60148972/1646932
    """
    _, layout = generated_chart
    assert layout.get("showlegend") == expected.get(
        "show"
    ), "Incorrect value for 'showlegend' field"


@pytest.mark.parametrize("updated_params, expected", legend_params_and_expected)
def test_layout_legend_position_success(generated_chart, expected):
    _, layout = generated_chart
    x, y = (layout["legend"].get("xanchor"), layout["legend"].get("yanchor"))
    assert [x, y] == expected.get(
        "pos"
    ), "Incorrect value for 'legend position' field"


@pytest.mark.parametrize("updated_params, expected", legend_params_and_expected)
def test_layout_legend_orientation_success(generated_chart, expected):
    _, layout = generated_chart
    orientation = layout["legend"].get("orientation")
    assert orientation == expected.get(
        "o"
    ), "Incorrect value for 'legend orientation' field"


# @pytest.mark.parametrize("updated_params, expected", [
#     ({'template': 'ggplot2'}, None)
# ])
# def test_layout_template_success(generated_chart, expected):
#     _, layout = generated_chart
#     breakpoint()
#     orientation = layout["legend"].get("orientation")
#     assert orientation == expected.get(
#         "o"
#     ), "Incorrect value for 'legend orientation' field"

# Color palette, opacity, fill opacity, smooth
