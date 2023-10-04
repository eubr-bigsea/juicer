from io import StringIO
import sys
from juicer.meta.meta_minion import MetaMinion
from juicer.workflow.workflow import Workflow
from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SortOperation
import pytest
from juicer.transpiler import GenerateCodeParams
from mock import patch, MagicMock
from .fixtures import * # Must be * in order to import fixtures
import hashlib

config = {
        'juicer': {
            'auditing': False,
            'services':{
                'limonero': {
                    'url': 'http://localhost',
                    'auth_token': 111
                }
            }

        }
    }
@patch('juicer.service.limonero_service.get_data_source_info')
@patch('juicer.workflow.workflow.Workflow._get_operations')
def test_sample(mocked_ops: MagicMock, limonero: MagicMock, 
        sample_workflow: dict, builder_params: any):

    mocked_ops.side_effect = mock_get_operations
    limonero.side_effect = mock_get_datasource
    for k in ['type', 'fraction', 'seed', 'value']:
        print(k, builder_params.sample.parameters.get(k))

    builder_params.sample.parameters['seed'] = 7777
    builder_params.sample.parameters['value'] = 30000
    print(builder_params.sample.model_builder_code())

@patch('juicer.service.limonero_service.get_data_source_info')
@patch('juicer.workflow.workflow.Workflow._get_operations')
def test_builder_params(mocked_ops: MagicMock, limonero: MagicMock, 
        sample_workflow: dict):
    
    mocked_ops.side_effect = mock_get_operations
    limonero.side_effect = mock_get_datasource

    loader = Workflow(sample_workflow, config, lang='en')

    instances = loader.workflow['tasks']

    minion = MetaMinion(None, config=config, workflow_id=sample_workflow['id'],
                        app_id=sample_workflow['id'])

    job_id = 1
    opt = GenerateCodeParams(loader.graph, job_id, None, {}, 
        ports={}, state={}, task_hash=hashlib.sha1(), 
        workflow=loader.workflow,
        tasks_ids=list(loader.graph.nodes.keys()))
    instances, _ = minion.transpiler.get_instances(opt)

    builder_params = minion.transpiler.prepare_model_builder_parameters(ops=instances.values())

    print(dir(builder_params))
    print(builder_params.read_data.model_builder_code())
    print(builder_params.sample.model_builder_code())
    print(builder_params.split.model_builder_code())
    print(builder_params.evaluator.model_builder_code())
    print(builder_params.features.model_builder_code())
    print(builder_params.reduction.model_builder_code())
    print(builder_params.grid.model_builder_code())
    
@patch('juicer.service.limonero_service.get_data_source_info')
@patch('juicer.workflow.workflow.Workflow._get_operations')
def test_generate_run_code_success(mocked_ops: MagicMock, limonero: MagicMock,
                      sample_workflow: dict):

    job_id = 1
    mocked_ops.side_effect = mock_get_operations
    limonero.side_effect = mock_get_datasource

    loader = Workflow(sample_workflow, config, lang='en')

    loader.handle_variables({'job_id': job_id})
    out = StringIO()

    minion = MetaMinion(None, config=config, workflow_id=sample_workflow['id'],
                        app_id=sample_workflow['id'])
    minion.transpiler.transpile(loader.workflow, loader.graph,
                                config, out, job_id,
                                persist=False)
    out.seek(0)
    code = out.read()

    # print(code, file=sys.stderr)

    result = util.execute(code, {'df': 'df'})
    # assert not result['out'].equals(test_df)
    # assert """out = df.sort_values(by=['sepalwidth'], ascending=[False])""" == \
    #        instance.generate_code()
