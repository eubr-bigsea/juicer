import json
import os
import pathlib
import pytest


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
    ]


@pytest.fixture(scope='function')
def sample_workflow() -> dict:

    module_dir = pathlib.Path(__file__).resolve().parent

    with open(module_dir / 'workflow_test_1.json') as f:
        return json.load(f)
