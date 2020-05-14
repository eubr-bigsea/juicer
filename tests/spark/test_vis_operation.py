# coding=utf-8
import ast
import uuid
from textwrap import dedent

import mock
import pytest

# noinspection PyUnresolvedReferences
from juicer.spark.vis_operation import AreaChartModel, BarChartModel, \
    ChartVisualization, DonutChartModel, HtmlVisualizationModel, \
    LineChartModel, MapModel, PieChartModel, ScatterPlotModel, \
    SummaryStatisticsModel, TableVisualizationModel, \
    VisualizationMethodOperation as Visu, \
    AreaChartOperation
from tests import compare_ast, format_code_comparison


class FakeDataType(object):
    def __init__(self, data_type):
        self.type = data_type

    # noinspection PyPep8Naming
    def jsonValue(self):
        t = {'StringType': 'str', 'IntegerType': 'int',
             'DatetimeType': 'date', 'FloatType': 'float'}
        return t[self.type]


class FakeDataframeAttribute(object):
    def __init__(self, name, data_type):
        self.name = name
        self.type = FakeDataType(data_type)

    # noinspection PyPep8Naming
    @property
    def dataType(self):
        return self.type


class FakeDataframe(object):
    def __init__(self, data):
        self.data = data

    def collect(self):
        return self.data

    @property
    def columns(self):
        return ','.join(self.data[0].keys())

    @property
    def schema(self):
        def get_data_type(value):
            if isinstance(value, str):
                return 'StringType'
            elif isinstance(value, int):
                return 'IntegerType'
            elif isinstance(value, float):
                return 'FloatType'

        return [FakeDataframeAttribute(name=k, data_type=get_data_type(v))
                for k, v in self.data[0].items()]


@pytest.fixture
def line_chart():
    pass


@pytest.fixture
def time_series_data():
    return FakeDataframe(
        [
            {"id": "1889", "value": 25.9},
            {"id": "1890", "value": 25.4},
            {"id": "1891", "value": 24.9},
            {"id": "1892", "value": 24.0},
            {"id": "1893", "value": 24.5},
            {"id": "1894", "value": 23.0},
            {"id": "1895", "value": 22.7},
            {"id": "1896", "value": 22.1},
            {"id": "1897", "value": 22.2},
            {"id": "1898", "value": 22.9},
        ]
    )


# noinspection PyShadowingNames
@pytest.fixture
def area_chart_model(time_series_data):
    expected = {
        'legend': {'isVisible': True, 'text': '{{name}}'},
        'title': '@@change@@',
        'tooltip': {
            'body': [
                "<span class='metric'>{{x}}</span>"
                "<span class='number'>{{y}}</span>"
            ], 'title': ['{{name}}']
        },
        'x': {'format': None, 'prefix': None, 'suffix': None, 'title': None,
              'type': 'number'},
        'y': {'format': None, 'prefix': None, 'suffix': None, 'title': None},
        'data': [{
            'values': [{u'x': item['id'], u'y': item['value']} for item in
                       time_series_data.collect()],
            'id': 'value',
            'name': 'value',
            'pointShape': 'diamond',
            'color': '#506FBB',
            'pointColor': '#506FBB',
            'pointSize': 3,
        }]
    }
    return expected


# noinspection PyShadowingNames
def atest_area_chart_model_success(time_series_data, area_chart_model):
    task_id = uuid.uuid4()
    params = {
        'x_attribute': ['id'],
    }
    model = AreaChartModel(data=time_series_data,
                           type_id=11,
                           task_id=task_id,
                           type_name='area-chart', title='Chart',
                           column_names='value',
                           orientation='landscape',
                           id_attribute='id', value_attribute='value',
                           params=params)

    area_chart_model['title'] = model.title

    assert model.get_icon() == 'fa-area-chart', 'Invalid icon'
    result = model.get_data()

    assert area_chart_model == result


def get_mocked_caipirinha_config(config, indentation=0):
    result = dedent("""
    config = {
        'juicer': {
            'services': {
                'limonero': {
                    'url': 'http://limonero:3321',
                    'auth_token': 'token'
                },
                'caipirinha': {
                    'url': 'http://caipirinha:3324',
                    'auth_token': 'token',
                    'storage_id': 1
                },
            }
        }
    }""")
    return result


# noinspection PyShadowingNames
def atest_area_chart_success(time_series_data):
    params = {
        Visu.TITLE_PARAM: 'Simple title 1',
        Visu.COLUMN_NAMES_PARAM: ['name, age, gender'],
        Visu.ORIENTATION_PARAM: 'landscape',
        Visu.ID_ATTR_PARAM: ['id'],
        Visu.VALUE_ATTR_PARAM: ['age'],
        'task': {
            'id': uuid.uuid4(),
        },
        'operation_id': 1,
        'operation_slug': 'area-chart',
        'user': {},
        'workflow_id': 17,
        'job_id': 100,
    }
    n_in = {'input data': 'input'}
    n_out = {}
    chart = AreaChartOperation(params, n_in, n_out)
    with mock.patch('juicer.spark.vis_operation.get_caipirinha_config',
                    get_mocked_caipirinha_config):
        code = chart.generate_code()

    expected_code = dedent("""
        from juicer.spark.vis_operation import AreaChartModel
        from juicer.util.dataframe_util import SimpleJsonEncoder as enc
        from juicer.service import caipirinha_service
        params = '{{}}'
        vis_task_1 = AreaChartModel(
            input, '{task_id}', '{operation_id}',
            '{operation_slug}', '{title}',
            {column_names},
            'landscape', {id_attribute}, {value_attribute},
            params=json.loads(params))
        config = {{
            'juicer': {{
                'services': {{
                    'limonero': {{
                        'url': 'http://limonero:3321',
                        'auth_token': 'token'
                    }},
                    'caipirinha': {{
                        'url': 'http://caipirinha:3324',
                        'auth_token': 'token',
                        'storage_id': 1
                    }},
                }}
            }}
        }}
        visualization = {{
           'job_id': '{job_id}',
           'task_id': vis_task_1.task_id,
           'title': vis_task_1.title ,
           'type': {{
               'id': vis_task_1.type_id,
               'name': vis_task_1.type_name
           }},
           'model': vis_task_1,
           'data': json.dumps(vis_task_1.get_data(), cls=enc, ignore_nan=True)
        }}
        caipirinha_service.new_visualization(
           config, {{}}, {workflow_id}, {job_id},
           '{task_id}',
           visualization, emit_event)""").format(
        task_id=params['task']['id'], **params)
    ast.parse(expected_code)
    result, msg = compare_ast(ast.parse(code), ast.parse(expected_code))
    assert result, msg + format_code_comparison(code, expected_code)
