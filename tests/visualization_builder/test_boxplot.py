import pytest
from juicer.scikit_learn.polars.vis_operation import VisualizationOperation
from tests.scikit_learn import util

# Boxplot

@pytest.fixture
def get_df():
    return util.tips_polars()

@pytest.fixture
def get_arguments():
    return {
        "parameters": {
            "type": "boxplot",
            "display_legend": "RIGHT",
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
                    "attribute": "time",
                    "displayLabel": "test",
                }
            ],
            "palette": [
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
                    "attribute": "total_bill",
                    #"aggregation": "MIN",
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
                },
                {
                    "attribute": "tip",
                    #"aggregation": "MIN",
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
                },
                {
                    "attribute": "size",
                    #"aggregation": "MIN",
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
            "input data": "tips",
        },
        "named_outputs": {"output data": "out"},
    }


#

@pytest.fixture()
def generated_chart(get_arguments, get_df):

    instance = VisualizationOperation(**get_arguments)
    vis_globals = dict(tips=get_df, emit_event=util.emit_event)
    code = "\n".join(
        [
            "import plotly.graph_objects as go",
            "import plotly.express as px",
            "from plotly.subplots import make_subplots",
            "import json",
            instance.generate_code(),
        ]
    )
    breakpoint()
    with open('lixo.py', 'w') as f:
        print(code, file=f)
    result = util.execute(code, vis_globals)
    generated_chart = result.get("d")
    data = generated_chart["data"]
    layout = generated_chart["layout"]

    return data, layout


# Data tests


# Test to verify the 'boxpoints' field
def test_boxplot_boxpoints(generated_chart):
    data, _ = generated_chart
    boxpoints = [item.get("boxpoints") for item in data]
    expected_boxpoints = ["all", "all"]
    assert boxpoints == expected_boxpoints, "Incorrect values for 'boxpoints' field"


# Test to verify the 'legendgroup' field
def test_boxplot_legendgroup(generated_chart):
    data, layout = generated_chart
    legendgroup = [item.get("legendgroup") for item in data]
    expected_legendgroup = ["Dinner", "Lunch"]
    assert (
        legendgroup == expected_legendgroup
    ), "Incorrect values for 'legendgroup' field"


# Test to verify the 'alignmentgroup' field
def test_boxplot_alignmentgroup(generated_chart):
    data, layout = generated_chart
    alignmentgroup = [item.get("alignmentgroup") for item in data]
    expected_alignmentgroup = ["True", "True"]
    assert (
        alignmentgroup == expected_alignmentgroup
    ), "Incorrect values for 'alignmentgroup' field"


# Test to verify the 'hovertemplate' field
def test_boxplot_hovertemplate(generated_chart):
    data, layout = generated_chart
    hovertemplate = [item.get("hovertemplate") for item in data]
    expected_hovertemplate = [
        "test=%{x}<br>min(total_bill)=%{y}<extra></extra>",
        "test=%{x}<br>min(total_bill)=%{y}<extra></extra>",
    ]
    assert (
        hovertemplate == expected_hovertemplate
    ), "Incorrect values for 'hovertemplate' field"


# Test to verify the 'marker' field
def test_boxplot_marker(generated_chart):
    data, layout = generated_chart
    markers = [item.get("marker") for item in data]
    expected_markers = [{"color": "#636efa"}, {"color": "#EF553B"}]
    assert markers == expected_markers, "Incorrect values for 'marker' field"


# Test to verify the 'name' field
def test_boxplot_name(generated_chart):
    data, layout = generated_chart
    names = [item.get("name") for item in data]
    expected_names = ["Dinner", "Lunch"]
    assert names == expected_names, "Incorrect values for 'name' field"


# Test to verify the 'notched' field
def test_boxplot_notched(generated_chart):
    data, layout = generated_chart
    notched_values = [item.get("notched") for item in data]
    expected_notched_values = [False, False]
    assert (
        notched_values == expected_notched_values
    ), "Incorrect values for 'notched' field"


# Test to verify the 'offsetgroup' field
def test_boxplot_offsetgroup(generated_chart):
    data, layout = generated_chart
    offsetgroups = [item.get("offsetgroup") for item in data]
    expected_offsetgroups = ["Dinner", "Lunch"]
    assert (
        offsetgroups == expected_offsetgroups
    ), "Incorrect values for 'offsetgroup' field"


# Test to verify the 'orientation' field
def test_boxplot_orientation(generated_chart):
    data, layout = generated_chart
    orientations = [item.get("orientation") for item in data]
    expected_orientations = ["v", "v"]
    assert (
        orientations == expected_orientations
    ), "Incorrect values for 'orientation' field"


# Test to verify the 'x' field
def test_boxplot_x(generated_chart):
    data, layout = generated_chart
    x_values = [item.get("x") for item in data]
    expected_x_values = [["Dinner"], ["Lunch"]]
    assert x_values == expected_x_values, "Incorrect values for 'x' field"


# Test to verify the 'xaxis' field
def test_boxplot_xaxis(generated_chart):
    data, layout = generated_chart
    xaxis_values = [item.get("xaxis") for item in data]
    expected_xaxis_values = ["x", "x"]
    assert xaxis_values == expected_xaxis_values, "Incorrect values for 'xaxis' field"


# Test to verify the 'y' field
def test_boxplot_y(generated_chart):
    data, layout = generated_chart
    y_values = [item.get("y") for item in data]
    expected_y_values = [[8.77], [9.55]]
    assert y_values == expected_y_values, "Incorrect values for 'y' field"


# Test to verify the 'yaxis' field
def test_boxplot_yaxis(generated_chart):
    data, layout = generated_chart
    yaxis_values = [item.get("yaxis") for item in data]
    expected_yaxis_values = ["y", "y"]
    assert yaxis_values == expected_yaxis_values, "Incorrect values for 'yaxis' field"


# Test to verify the 'quartilemethod' field
def test_boxplot_quartilemethod(generated_chart):
    data, layout = generated_chart
    util.save_chart(data, layout)
   # quartilemethods = [item.get("quartilemethod") for item in data]
   # expected_quartilemethods = ["exclusive", "exclusive"]
   # assert (
   #     quartilemethods == expected_quartilemethods
   # ), "Incorrect values for 'quartilemethod' field"
