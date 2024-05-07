import pytest

from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from tests.scikit_learn import util


@pytest.fixture(scope="session")
def get_df():
    return util.iris_polars()

@pytest.fixture
def get_arguments():
    return {
        "parameters": {
            "type": "pie",
            "display_legend": "AUTO",
            "x": [
                {
                    "binning": None,
                    "bins": 20,
                    "binSize": 10,
                    "emptyBins": "ZEROS",
                    "multiplier": None,
                    "decimal_places": 2,
                    "prefix": None,
                    "suffix": None,
                    "label": None,
                    "max_displayed": None,
                    "group_others": True,
                    "sorting": "NATURAL",
                    "attribute": "class",
                }
            ],
            "color_scale": [
                "#1F77B4",
                "#FF7F0E",
                "#2CA02C",
                "#D62728",
                "#9467BD",
                "#8C564B",
                "#E377C2",
                "#7F7F7F",
                "#BCBD22",
                "#17BECF",
            ],
            "y": [
                {
                    "attribute": "*",
                    "aggregation": "COUNT",
                    "compute": None,
                    "displayOn": "left",
                    "multiplier": None,
                    "decimal_places": 2,
                    "prefix": None,
                    "suffix": None,
                    "label": None,
                    "strokeSize": 0,
                    "stroke": None,
                    "color": None,
                    "marker": None,
                }
            ],
            "x_axis": {
                "lowerBound": None,
                "upperBound": None,
                "logScale": False,
                "display": True,
                "displayLabel": True,
                "label": None,
                "multiplier": None,
                "decimal_places": 2,
                "prefix": None,
                "suffix": None,
            },
            "y_axis": {
                "lowerBound": None,
                "upperBound": None,
                "logScale": False,
                "display": True,
                "displayLabel": True,
                "label": None,
                "multiplier": None,
                "decimal_places": 2,
                "prefix": None,
                "suffix": None,
            },
            "task_id": "0",
        },
        "named_inputs": {
            "input data": "iris",
        },
        "named_outputs": {"output data": "out"},
    }


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
    generated_chart = result.get("d")
    data = generated_chart["data"]
    layout = generated_chart["layout"]

    return data, layout


# Data tests
# Test to verify the 'domain' field
def test_data_domain(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    domain = data_entry.get("domain")
    assert domain is not None, "Field 'domain' not found in data"
    assert domain == {
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
    }, "Incorrect value for 'domain' field"


# Test to verify the 'hovertemplate' field
def test_data_hovertemplate(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    hovertemplate = data_entry.get("hovertemplate")
    assert hovertemplate is not None, "Field 'hovertemplate' not found in data"
    assert (
        hovertemplate == "class=%{label}<br>count(*)=%{value}<extra></extra>"
    ), "Incorrect value for 'hovertemplate' field"


# Test to verify the 'labels' field
def test_data_labels(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    labels = data_entry.get("labels")
    assert labels is not None, "Field 'labels' not found in data"
    assert labels == [
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica",
    ], "Incorrect value for 'labels' field"


# Test to verify the 'legendgroup' field
def test_data_legendgroup(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    legendgroup = data_entry.get("legendgroup")
    assert legendgroup == "", "Incorrect value for 'legendgroup' field"


# Test to verify the 'showlegend' field
def test_data_showlegend(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    showlegend = data_entry.get("showlegend")
    assert showlegend is not None, "Field 'showlegend' not found in data"
    assert showlegend is True, "Incorrect value for 'showlegend' field"


# Test to verify the 'values' field
def test_data_values(generated_chart):
    data, layout = generated_chart
    data_entry = data[0]
    values = data_entry.get("values")
    assert values is not None, "Field 'values' not found in data"
    assert values == [50.0, 50.0, 50.0], "Incorrect value for 'values' field"


# Test to verify the 'legend' field
def test_layout_legend(generated_chart):
    data, layout = generated_chart
    legend = layout.get("legend")
    assert legend is not None, "Field 'legend' not found in layout"
    assert legend == {"tracegroupgap": 0}, "Incorrect value for 'legend' field"


# Test to verify the 'extendpiecolors' field
def test_layout_extendpiecolors(generated_chart):
    data, layout = generated_chart
    extendpiecolors = layout.get("extendpiecolors")
    assert extendpiecolors is not None, "Field 'extendpiecolors' not found in layout"
    assert extendpiecolors is True, "Incorrect value for 'extendpiecolors' field"
