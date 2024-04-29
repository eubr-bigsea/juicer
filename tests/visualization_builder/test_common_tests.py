import pytest

from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from tests.scikit_learn import util
# Common tests


@pytest.fixture(scope="session")
def get_df():
    return util.titanic_polars()


params_and_expected = [
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


def emit_event(*args, **kwargs):
    print(args, kwargs)


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


# Test to verify the 'margin' field
def test_layout_margin(generated_chart):
    _, layout = generated_chart
    layout_margin = layout.get("margin")
    assert (
        layout_margin is not None
    ), "Field 'margin' not found in the layout object"
    expected_margin = {"t": 30, "l": 30, "r": 30, "b": 30}
    assert (
        layout_margin == expected_margin
    ), "Incorrect values for the 'margin' field"


# Test for verifying the 'hovertemplate' field
def test_data_hovertemplate(generated_chart):
    data, _ = generated_chart
    sunburst_data = data[0]
    hovertemplate = sunburst_data.get("hovertemplate")
    assert (
        hovertemplate is not None
    ), "Field 'hovertemplate' not found in data object"


@pytest.mark.parametrize("updated_params, expected", params_and_expected)
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


@pytest.mark.parametrize("updated_params, expected", params_and_expected)
def test_layout_legend_position_success(generated_chart, expected):
    _, layout = generated_chart
    x, y = (layout["legend"].get("xanchor"), layout["legend"].get("yanchor"))
    assert [x, y] == expected.get(
        "pos"
    ), "Incorrect value for 'legend position' field"


@pytest.mark.parametrize("updated_params, expected", params_and_expected)
def test_layout_legend_orientation_success(generated_chart, expected):
    _, layout = generated_chart
    orientation = layout["legend"].get("orientation")
    assert orientation == expected.get(
        "o"
    ), "Incorrect value for 'legend orientation' field"
