import pytest

from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from tests.scikit_learn import util


@pytest.fixture(scope="session")
def get_df():
    return util.iris_polars()


@pytest.fixture(scope='function')
def get_arguments(updated_params):
    result = {
        "parameters": {
            "display_legend": "AUTO",
            "smoothing": False,
            "number_format": ",.",
            "palette": [
                "#17BECF",
                "#9467BD",
                "#E377C2",
                "#D62728",
                "#FF0000",
                "#FFFF00",
            ],
            "fill_opacity": 0.5,
            "x": [
                {
                    "bins": 20,
                    "binSize": 10,
                    "max_displayed": 5,
                    "group_others": True,
                    "sorting": "NATURAL",
                    "attribute": "sepalwidth",
                }
            ],
            "y": [
                {
                    "attribute": "sepallength",
                    "aggregation": "MIN",
                    "label": "Min sepal length",
                    "xstrokeSize": 4,
                    "xstroke": "dashdot",
                    "xcolor": "#000000",
                    "enabled": True,
                    "xcustom_color": True,
                    "xline_color": "#4d43d0",
                    "xline_size": 5,
                },
                {
                    "attribute": "sepallength",
                    "aggregation": "MAX",
                    "label": "Max sepal length",
                    "xstrokeSize": 4,
                    "xstroke": "dashdot",
                    "xcolor": "#FF00FF",
                    "enabled": True,
                    "xcustom_color": True,
                    "xline_color": "#4d43d0",
                    "xline_size": 5,
                },
            ],
            "type": "line",
            "x_axis": {
            },
            "y_axis": {
            },
            "auto_margin": True,
            "task_id": "ababababa",
        },
        "named_inputs": {
            "input data": "iris",
        },
        "named_outputs": {"output data": "out"},
    }
    for p, v in updated_params.items():
        util.update_dict(result["parameters"], p, v)
    return result


@pytest.fixture
def generated_chart(get_arguments, get_df):
    instance = VisualizationOperation(**get_arguments)
    vis_globals = dict(iris=get_df, emit_event=util.emit_event)
    code = instance.generate_code()
    code = "\n".join(
        [
            "import plotly.graph_objects as go",
            "import plotly.express as px",
            "import json",
            code,
        ]
    )
    #breakpoint()
    result = util.execute(code, vis_globals)
    generated_chart_code = result.get("d")
    data = generated_chart_code["data"]
    layout = generated_chart_code["layout"]

    return data, layout


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        ({"type": "line", "y_axis.logScale": False}, None),
        ({"type": "line", "y_axis.logScale": True}, "log"),
        (
            {"type": "pie", "y_axis.logScale": True},
            None,
        ),  # pie doesn't have axis
        # ({"type": "bar",'y_axis.logScale': True}, 'log'),
        # ({"type": "stacked-bar",'y_axis.logScale': True}, 'log'),
        # ({"type": "stacked-area-100",'y_axis.logScale': True}, 'log'),
        # ({"type": "stacked-area",'y_axis.logScale': True}, 'log'),
        # ({"type": "scatter",'y_axis.logScale': True}, 'log'),
    ],
)
def test_y_axis_log_scale_success(generated_chart, expected):
    _, layout = generated_chart
    assert layout["yaxis"].get("type") == expected


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        (
            {"type": "pie", "x_axis.logScale": True},
            None,
        ),  # pie doesn't have axis
        ({"type": "scatter", "x_axis.logScale": True}, "log"),
    ],
)
def test_x_axis_log_scale_success(generated_chart, expected):
    _, layout = generated_chart
    assert layout["xaxis"].get("type") == expected


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        (
            {"type": "pie", "x_axis.label": "Test chart"},
            None,
        ),  # pie doesn't have axis
        (
            {"type": "stacked-bar", "x_axis.label": "Test chart"},
            {"text": "Test chart"},
        ),
    ],
)
def test_x_axis_title_success(generated_chart, expected):
    _, layout = generated_chart
    assert layout["xaxis"].get("title") == expected


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        (
            {"type": "pie", "y_axis.label": "Test chart Y"},
            None,
        ),  # pie doesn't have axis
        (
            {"type": "bar", "y_axis.label": "Test chart Y"},
            {"text": "Test chart Y"},
        ),
    ],
)
def test_y_axis_title_success(generated_chart, expected):
    _, layout = generated_chart
    assert layout["yaxis"].get("title") == expected


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        (
            {
                "type": "stacked-area-100",
                "x_axis.displayLabel": True,
                "x_axis.label": "Test chart X",
            },
            {"text": "Test chart X"},
        ),
        (
            {
                "type": "stacked-area-100",
                "x_axis.displayLabel": False,
                "x_axis.label": "Test chart X",
            },
            {},
        ),
        (
            {
                "type": "stacked-area-100",
                "x_axis.displayLabel": True,
                "x_axis.label": None,
            },
            {"text": "sepalwidth"},
        ),
    ],
)
def test_x_axis_display_title_success(generated_chart, expected):
    _, layout = generated_chart
    assert layout["xaxis"].get("title") == expected


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        (
            {
                "type": "stacked-area",
                "y_axis.displayLabel": True,
                "y_axis.label": "Test chart Y",
            },
            {"text": "Test chart Y"},
        ),
        (
            {
                "type": "stacked-area",
                "y_axis.displayLabel": False,
                "y_axis.label": "Test chart Y",
            },
            {},
        ),
        (
            {
                "type": "stacked-area",
                "y_axis.displayLabel": True,
                "y_axis.label": "",
            },
            {"text": "value"}, # Don't know why this is the default
        ),
    ],
)
def test_y_axis_display_title_success(generated_chart, expected):
    _, layout = generated_chart
    assert layout["yaxis"].get("title") == expected

@pytest.mark.parametrize(
    "updated_params, expected",
    [
        (
            {"type": "line", "x_axis.display": True},
            True,
        ),
        (
            {"type": "line"},
            True,
        ),
        (
            {"type": "line", "x_axis.display": False},
           False,
        ),
    ],
)
def test_x_axis_display_success(generated_chart, expected):
    _, layout = generated_chart
    assert layout["xaxis"].get("visible") == expected


@pytest.mark.parametrize(
    "updated_params, expected",
    [
        (
            {"type": "line", "y_axis.display": True},
            True,
        ),
        (
            {"type": "line"},
            True,
        ),
        (
            {"type": "line", "y_axis.display": False},
           False,
        ),
    ],
)
def test_y_axis_display_success(generated_chart, expected):
    _, layout = generated_chart
    assert layout["yaxis"].get("visible") == expected

@pytest.mark.parametrize(
    "updated_params, expected",
    [
        (
            {"type": "line", "x_axis.prefix": '^^', 'x_axis.suffix': '$$'},
            ("^^", "$$"),
        ),
        (
            {"type": "line"},
            (None, None),
        )
    ],
)
def test_x_axis_prefix_suffix_success(generated_chart, expected):
    _, layout = generated_chart

    assert layout["xaxis"].get("tickprefix") == expected[0]
    assert layout["xaxis"].get("ticksuffix") == expected[1]

@pytest.mark.parametrize(
    "updated_params, expected",
    [
        (
            {"type": "line", "y_axis.prefix": '^^', 'y_axis.suffix': '$$'},
            ("^^", "$$"),
        ),
        (
            {"type": "line"},
            (None, None),
        )
    ],
)
def test_y_axis_prefix_suffix_success(generated_chart, expected):
    _, layout = generated_chart
    assert layout["yaxis"].get("tickprefix") == expected[0]
    assert layout["yaxis"].get("ticksuffix") == expected[1]

# TODO: implement test for the range limits. There is a bug in plotly
# See https://github.com/plotly/plotly.py/issues/3634