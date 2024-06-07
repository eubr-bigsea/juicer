import pytest

from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from tests.scikit_learn import util


@pytest.fixture
def get_df():
    return util.titanic_polars()


@pytest.fixture
def get_arguments():
    return {
        "parameters": {
            "type": "sunburst",
            "display_legend": "HIDE",
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
                    "enabled": True,
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


# Test for verifying the 'branchvalues' field
def test_data_branchvalues(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]
    branchvalues = sunburst_data.get("branchvalues")
    assert (
        branchvalues is not None
    ), "Field 'branchvalues' not found in data object"
    assert branchvalues == "total", "Incorrect value for 'branchvalues' field"


# Test for verifying the 'customdata' field
def test_data_customdata(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]
    customdata = sunburst_data.get("customdata")
    assert customdata is not None, "Field 'customdata' not found in data object"
    expected_customdata = [[323.0], [277.0], [709.0]]
    assert (
        customdata == expected_customdata
    ), "Incorrect value for 'customdata' field"


# Test for verifying the 'domain' field
def test_data_domain(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]
    domain = sunburst_data.get("domain")
    assert domain is not None, "Field 'domain' not found in data object"


# Test for verifying the 'ids' field
def test_data_ids(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]
    ids = sunburst_data.get("ids")
    assert ids is not None, "Field 'ids' not found in data object"
    expected_ids = ["1st", "2nd", "3rd"]
    assert ids == expected_ids, "Incorrect values for 'ids' field"


# Test for verifying the 'labels' field
def test_data_labels(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]
    labels = sunburst_data.get("labels")
    assert labels is not None, "Field 'labels' not found in data object"
    expected_labels = ["1st", "2nd", "3rd"]
    assert labels == expected_labels, "Incorrect values for 'labels' field"


# Test for verifying the 'marker' field
def test_data_marker(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]
    marker = sunburst_data.get("marker")
    assert marker is not None, "Field 'marker' not found in data object"


# Test for verifying the 'values' field
def test_data_values(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]
    values = sunburst_data.get("values")
    assert values is not None, "Field 'values' not found in data object"
    expected_values = [323.0, 277.0, 709.0]
    assert values == expected_values, "Incorrect values for 'values' field"


# Test for verifying the 'type' field
def test_data_type(generated_chart):
    data, layout = generated_chart
    sunburst_data = data[0]
    chart_type = sunburst_data.get("type")
    assert chart_type is not None, "Field 'type' not found in data object"
    assert chart_type == "sunburst", "Incorrect value for 'type' field"


# Layout tests


# Test for verifying the 'template' field
def test_layout_template(generated_chart):
    data, layout = generated_chart
    layout_template = layout.get("template")
    assert (
        layout_template is not None
    ), "Field 'template' not found in layout object"
    expected_template = {"data": {"scatter": [{"type": "scatter"}]}}
    assert (
        layout_template == expected_template
    ), "Incorrect values for 'template' field"


# Test for verifying the 'coloraxis' field
def test_layout_coloraxis(generated_chart):
    data, layout = generated_chart
    layout_coloraxis = layout.get("coloraxis")
    assert (
        layout_coloraxis is not None
    ), "Field 'coloraxis' not found in layout object"
    expected_coloraxis = {
        "colorbar": {"title": {"text": "count(*)"}},
        "colorscale": [
            [0.0, "#000000"],
            [0.25, "#e60000"],
            [0.5, "#e6d200"],
            [0.75, "#ffffff"],
            [1.0, "#a0c8ff"],
        ],
    }
    assert (
        layout_coloraxis == expected_coloraxis
    ), "Incorrect values for 'coloraxis' field"


# Test for verifying the 'legend' field
def test_layout_legend(generated_chart):
    data, layout = generated_chart
    layout_legend = layout.get("legend")
    assert layout_legend is not None, "Field 'legend' not found in layout object"
    expected_legend = {"tracegroupgap": 0}
    assert (
        layout_legend == expected_legend
    ), "Incorrect values for 'legend' field"


# Test for verifying the 'xaxis' field
def test_layout_xaxis(generated_chart):
    data, layout = generated_chart
    layout_xaxis = layout.get("xaxis")
    assert layout_xaxis is not None, "Field 'xaxis' not found in layout object"
    expected_xaxis = {"categoryorder": "trace"}
    assert layout_xaxis == expected_xaxis, "Incorrect values for 'xaxis' field"
