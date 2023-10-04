from io import StringIO
import sys
from juicer.meta.meta_minion import MetaMinion
from juicer.workflow.workflow import Workflow
from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import SortOperation
import pytest
from mock import patch, MagicMock
from .fixtures import * # Must be * in order to import fixtures


@patch('juicer.service.limonero_service.get_data_source_info')
@patch('juicer.workflow.workflow.Workflow._get_operations')
def test_sort_success(mocked_ops: MagicMock, limonero: MagicMock,
                      sample_workflow: dict):

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

    print(code, file=sys.stderr)

    # df = util.iris(['sepallength', 'sepalwidth'], size=150)
    # test_df = df.copy()
    # arguments = {
    #     'parameters': {
    #         'attributes': [{'attribute': 'sepalwidth',
    #                         'f': 'desc'}]},
    #     'named_inputs': {
    #         'input data': 'df',
    #     },
    #     'named_outputs': {
    #         'output data': 'out'
    #     }
    # }
    # instance = SortOperation(**arguments)
    # result = util.execute(util.get_complete_code(instance), {'df': df})
    # assert not result['out'].equals(test_df)
    # assert """out = df.sort_values(by=['sepalwidth'], ascending=[False])""" == \
    #        instance.generate_code()
