import json
import os
import pathlib
import hashlib
import pytest

from juicer.meta.meta_minion import MetaMinion
from juicer.workflow.workflow import Workflow
from mock import patch, MagicMock
from juicer.transpiler import GenerateCodeParams

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


def mock_get_datasource(*args):
    module_dir = pathlib.Path(__file__).resolve().parent.parent
    iris = module_dir / 'data' / 'iris.csv.gz'
    return {
        'storage': {'id': 1, 'name': 'Local'},
        'format': 'CSV',
        'url': f'file://{iris}',
        'is_first_line_header': True,
        'infer_schema': 'FROM_LIMONERO',
        'attributes': [
            {'name': 'sepallength', 'type': 'FLOAT'},
            {'name': 'sepalwidth', 'type': 'FLOAT'},
            {'name': 'petallength', 'type': 'FLOAT'},
            {'name': 'petalwidth', 'type': 'FLOAT'},
            {'name': 'class', 'type': 'CHARACTER'},
        ]
    }


def mock_get_operations(*args):
    return [
        {'id': 2100, 'slug': 'read-data', 'ports': []},
        {'id': 2110, 'slug': 'sample', 'ports': []},
        {'id': 2351, 'slug': 'evaluator', 'ports': []},
        {'id': 2350, 'slug': 'split', 'ports': []},
        {'id': 2352, 'slug': 'features-reduction', 'ports': []},
        {'id': 2353, 'slug': 'grid', 'ports': []},
        {'id': 2354, 'slug': 'features', 'ports': []},
        {'id': 2355, 'slug': 'k-means', 'ports': []},
        {'id': 2356, 'slug': 'gaussian-mix', 'ports': []},
        {'id': 2357, 'slug': 'decision-tree-classifier', 'ports': []},
        {'id': 2358, 'slug': 'gbt-classifier', 'ports': []},
        {'id': 2359, 'slug': 'naive-bayes', 'ports': []},
        {'id': 2360, 'slug': 'perceptron', 'ports': []},
        {'id': 2361, 'slug': 'random-forest-classifier', 'ports': []},
        {'id': 2362, 'slug': 'logistic-regression', 'ports': []},
        {'id': 2363, 'slug': 'svm', 'ports': []},
        {'id': 2364, 'slug': 'linear-regression', 'ports': []},
        {'id': 2365, 'slug': 'isotonic-regression', 'ports': []},
        {'id': 2366, 'slug': 'gbt-regressor', 'ports': []},
        {'id': 2367, 'slug': 'random-forest-regressor', 'ports': []},
        {'id': 2368, 'slug': 'generalized-linear-regressor', 'ports': []},
        {'id': 2369, 'slug': 'decision-tree-regressor', 'ports': []}
    ]


@pytest.fixture(scope='function')
def sample_workflow() -> dict:

    module_dir = pathlib.Path(__file__).resolve().parent

    with open(module_dir / 'workflow_test_1.json') as f:
        return json.load(f)

@pytest.fixture(scope='function')
def builder_params(sample_workflow: dict):
    """
    This fixture mocks methods used to interact with other Lemonade
    services (e.g. Limonero and Tahiti). It also returns an object
    organized according to the Model Builder template.
    """
    with patch('juicer.workflow.workflow.Workflow._get_operations', 
        return_value=mock_get_operations()):
        with patch('juicer.service.limonero_service.get_data_source_info',
            return_value=mock_get_datasource()):

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
            yield builder_params
