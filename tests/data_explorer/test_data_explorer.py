from io import StringIO
import uuid

import pytest

from juicer.meta.meta_minion import MetaMinion
from juicer.meta.operations import ReadDataOperation, SampleOperation, TransformOperation
from juicer.meta.transpiler import (
    MetaTranspiler)
from juicer.workflow.workflow import Workflow
from mock import patch
from .fixtures import *  # Must be * in order to import fixtures


# region Common tests
@pytest.mark.parametrize(
    'display_order, is_last', [(0, False), (1, True), (5, True)])
def test_task_positioning_and_display_sample(display_order: int, is_last: bool,
                                             transpiler: MetaTranspiler):
    task_id = "2222222-22222-fffff"
    task_name = "data reader"
    params = {
        "data_source": 1000,
        "display_sample": "0",
        "task": {"id": task_id, "name": task_name,
                 "display_order": display_order,
                 "enabled": True},
        "workflow": {"type": "DATA_EXPLORER"},
        "transpiler": transpiler
    }
    op = ReadDataOperation(params, {}, {'output data': 'df'})
    op.set_last(is_last)
    result = json.loads(op.generate_code())
    assert result['id'] == f'{task_id}-0'  # add suffix -0 to all tasks
    assert result['environment'] == 'DESIGN'
    assert result['left'] == (display_order % 4) * 250 + 100
    assert result['top'] == (display_order // 4) * 150 + 100
    assert result['z_index'] == 10
    assert result['display_order'] == display_order
    assert result['operation']['id'] == 18  # Data reader
    assert result['forms'].get(
        'display_sample', {}).get('value') == '1' if is_last else '0'

# endregion
# region Test Read Data and Sample Operations


def old_test_data_reader(titanic_workflow: dict):
    job_id = 1
    with patch('juicer.workflow.workflow.Workflow._get_operations',
               return_value=mock_get_operations()):
        with patch('juicer.service.limonero_service.get_data_source_info',
                   return_value=mock_get_datasource()):
            loader = Workflow(titanic_workflow, config, lang='en')

    params = {
        "data_source": {
            "value": 1000,
        },
        "display_sample": {
            "value": "0"
        },
    }
    instances = loader.workflow['tasks']

    loader.handle_variables({'job_id': job_id})
    out = StringIO()
    minion = MetaMinion(None, config=config,
                        workflow_id=titanic_workflow['id'],
                        app_id=titanic_workflow['id'])

    minion.transpiler.transpile(loader.workflow, loader.graph,
                                config, out, job_id,
                                persist=False)
    out.seek(0)
    code = out.read()
    print(code)


def test_data_reader_success(transpiler: MetaTranspiler):
    task_id = "2222222-22222-fffff"
    task_name = "data reader"
    display_order = 0

    params = {
        "data_source": 1000,
        "display_sample": "0",
        "task": {"id": task_id, "name": task_name,
                 "display_order": display_order,
                 "enabled": True},
        "workflow": {"type": "DATA_EXPLORER"},
        "transpiler": transpiler
    }
    op = ReadDataOperation(params, {}, {'output data': 'df'})
    result = json.loads(op.generate_code())
    assert result['operation']['id'] == 18  # Data reader
    assert result['forms']['data_source']['value'] == params['data_source']


def test_data_reader_missing_data_source_failure(transpiler: MetaTranspiler):
    task_id = "aaaa-2cdfdd-fffff"
    task_name = "data reader"
    display_order = 0

    params = {
        # "data_source": 1000, //Missing
        "display_sample": "0",
        "task": {"id": task_id, "name": task_name,
                 "display_order": display_order,
                 "enabled": True},
        "workflow": {"type": "DATA_EXPLORER"},
        "transpiler": transpiler
    }
    with pytest.raises(ValueError) as ve:
        ReadDataOperation(params, {}, {'output data': 'df'})

    assert 'Missing required parameter: data_source' in str(ve)


def test_sample_success(transpiler: MetaTranspiler):
    task_id = str(uuid.uuid4())
    task_name = "sample"
    display_order = 1

    params = {
        "type": "head",
        "value": 1000,
        "seed": 42,
        "fraction": .8,
        "display_sample": "1",
        "task": {"id": task_id, "name": task_name,
                 "display_order": display_order,
                 "enabled": True},
        "workflow": {"type": "DATA_EXPLORER"},
        "transpiler": transpiler
    }
    op = SampleOperation(params, {}, {'output data': 'df'})
    result = json.loads(op.generate_code())
    # Sample delegates parameters validation to target operation
    assert result['operation']['id'] == op.TARGET_OP
    assert result['forms']['type']['value'] == params['type']
    assert result['forms']['value']['value'] == params['value']
    assert result['forms']['fraction']['value'] == params['fraction']
    assert result['forms']['seed']['value'] == params['seed']


# endregion
# region Transform operation
@pytest.mark.parametrize(
    'slug, lixo', [('extract-numbers', False), ])
def test_transform_string_functions_success(
        slug: str, lixo: any, transpiler: MetaTranspiler):
    task_id = str(uuid.uuid4())
    task_name = "sample"
    display_order = 1

    params = {
        "attributes": ["boat", "embarked"],
        "display_sample": "1",
        "task": {"id": task_id, "name": task_name,
                 "display_order": display_order,
                 "enabled": True,
                 "operation": {"slug": slug}},
        "workflow": {"type": "DATA_EXPLORER"},
        "transpiler": transpiler
    }
    op = TransformOperation(params, {}, {'output data': 'df'})
    result = json.loads(op.generate_code())
    assert result['operation']['id'] == op.TARGET_OP
    print(result['forms']['expression'])


def test_transform_missing_attributes_failure(transpiler: MetaTranspiler):
    task_id = str(uuid.uuid4())
    task_name = "sample"
    display_order = 1

    params = {
        "type": "head",
        "value": 1000,
        "seed": 42,
        "fraction": .8,
        "display_sample": "1",
        "task": {"id": task_id, "name": task_name,
                 "display_order": display_order,
                 "enabled": True,
                 "operation": {"slug": "substring"}},
        "workflow": {"type": "DATA_EXPLORER"},
        "transpiler": transpiler
    }
    with pytest.raises(ValueError) as ve:
        TransformOperation(params, {}, {'output data': 'df'})
    assert 'Missing required parameter: attributes' in str(ve)
# endregion
