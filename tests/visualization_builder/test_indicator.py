import pytest

from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from tests.scikit_learn import util


@pytest.fixture(scope="session")
def get_df():
    return util.titanic_polars()


@pytest.fixture(scope="function")
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
                    "attribute": "pclass",
                }
            ],
            "y": [
                {
                    "attribute": "fare",
                    "aggregation": "max",
                    "label": "Test",
                    "xstrokeSize": 4,
                    "xstroke": "dashdot",
                    "xcolor": "#000000",
                    "enabled": True,
                    "xcustom_color": True,
                    "xline_color": "#4d43d0",
                    "xline_size": 5,
                }
            ],
            "type": "line",
            "x_axis": {},
            "y_axis": {},
            "auto_margin": True,
            "task_id": "ababababa",
        },
        "named_inputs": {
            "input data": "titanic",
        },
        "named_outputs": {"output data": "out"},
    }
    for p, v in updated_params.items():
        util.update_dict(result["parameters"], p, v)
    return result


@pytest.fixture
def generated_chart(get_arguments, get_df):
    instance = VisualizationOperation(**get_arguments)
    vis_globals = dict(titanic=get_df, emit_event=util.emit_event)
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
        ({"type": "indicator", "x": []}, None),
    ],
)
def test_indicator_no_x_success(generated_chart, expected):
    _, layout = generated_chart
    #breakpoint()
    #assert layout["yaxis"].get("type") == expected
    #util.save_chart(_, layout)

