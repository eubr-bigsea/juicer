import json
import datetime
import pathlib
from typing import Dict

import pandas as pd
from mock import patch

from juicer.scikit_learn.data_operation import DataReaderOperation
from tests.scikit_learn import util
from tests.service import fake_req
import pytest
from mock.mock import MagicMock
# DataReader
#
DATA_SOURCE_ID = 1000000
CONFIGURATION: Dict = {
    'juicer': {
        'services': {
            'limonero': {
                'url': 'http://localhost:8888',
                'auth_token': 'token'
            }
        }
    }
}


def _get_mock_payload(complement: Dict[str, any] = None) -> Dict[str, any]:
    payload = {
        'storage': {'id': 1, 'name': 'Local'},
        'format': 'CSV',
        'url': f'file://{pathlib.Path().absolute()}/tests/data/iris.csv.gz',
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
    if complement:
        payload.update(complement)
    return payload


def test_data_reader_missing_data_source_fail():
    arguments = {
        'parameters': {},
        'named_inputs': {},
        'named_outputs': {'output data': 'out'}
    }
    with pytest.raises(ValueError) as val_err:
        DataReaderOperation(**arguments)

    assert "Parameter 'data_source' must be informed for task" \
           " DataReaderOperation" in str(val_err.value)


@patch('juicer.service.limonero_service.get_data_source_info')
def test_data_reader_missing_url_in_limonero_fail(mocked_get: MagicMock):
    arguments = {
        'parameters': {
            'data_source': DATA_SOURCE_ID, 'workflow': {'data_source_cache': {}},
            'configuration': CONFIGURATION,
        },
        'named_inputs': {},
        'named_outputs': {'output data': 'out'}
    }

    mocked_get.side_effect = lambda *_: _get_mock_payload({'url': None})
    with pytest.raises(ValueError) as val_err:
        DataReaderOperation(**arguments)

    assert ("Incorrect data source configuration (empty url)" 
        in str(val_err.value))


@patch('juicer.service.limonero_service.get_data_source_info')
def test_data_reader_supports_cache_success(mocked_get: MagicMock):
    now = datetime.datetime.now()
    arguments = {
        'parameters': {
            'data_source': DATA_SOURCE_ID,
            'configuration': CONFIGURATION,
            'workflow': {'data_source_cache': {}},
            'execution_date': now,
        },
        'named_inputs': {},
        'named_outputs': {'output data': 'out'}
    }
    # Mock request to Limonero
    payload = _get_mock_payload(
        {'updated': (now - datetime.timedelta(minutes=1)).isoformat()})
    mocked_get.side_effect = lambda *_: payload
    instance = DataReaderOperation(**arguments)

    assert instance.supports_cache


@patch('juicer.service.limonero_service.get_data_source_info')
def test_data_reader_local_csv_limonero_success(mocked_get: MagicMock):
    df: pd.DataFrame = util.iris()

    arguments = {
        'parameters': {
            'data_source': DATA_SOURCE_ID,
            'configuration': CONFIGURATION,
            'workflow': {'data_source_cache': {}}
        },
        'named_inputs': {},
        'named_outputs': {'output data': 'out'}
    }
    # Mock request to Limonero
    mocked_get.side_effect = lambda *_: _get_mock_payload()

    instance = DataReaderOperation(**arguments)
    result: Dict[str, any] = util.execute(instance.generate_code(),  {'df': df})

    assert result['out'].compare(df).all().all()

@patch('juicer.service.limonero_service.get_data_source_info')
def test_data_reader_local_text_limonero_success(mocked_get: MagicMock):
    arguments = {
        'parameters': {
            'data_source': DATA_SOURCE_ID,
            'configuration': CONFIGURATION,
            'workflow': {'data_source_cache': {}}
        },
        'named_inputs': {},
        'named_outputs': {'output data': 'out'}
    }
    # Mock request to Limonero
    mocked_get.side_effect = lambda *_: _get_mock_payload({
        'format': 'TEXT', 
        'url': f'file://{pathlib.Path().absolute()}/requirements.txt',
        'attributes': [{'name': 'text', 'type': 'CHARACTER'}]
    })

    instance = DataReaderOperation(**arguments)
    result: Dict[str, any] = util.execute(instance.generate_code(), {})

    assert result['out'].columns == ['value']

@patch('juicer.service.limonero_service.get_data_source_info')
def test_data_reader_local_parquet_limonero_success(mocked_get: MagicMock):
    df: pd.DataFrame = util.iris()
    arguments = {
        'parameters': {
            'data_source': DATA_SOURCE_ID,
            'configuration': CONFIGURATION,
            'workflow': {'data_source_cache': {}}
        },
        'named_inputs': {},
        'named_outputs': {'output data': 'out'}
    }
    # Mock request to Limonero
    mocked_get.side_effect = lambda *_: _get_mock_payload({
        'format': 'PARQUET', 
        'url': f'file://{pathlib.Path().absolute()}/tests/data/iris.parquet',
    })

    instance = DataReaderOperation(**arguments)
    result: Dict[str, any] = util.execute(instance.generate_code(), {})
    assert result['out'].compare(df).all().all()