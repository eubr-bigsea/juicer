from tests.scikit_learn import util
from juicer.scikit_learn.data_operation import SaveOperation
from .test_data_reader import CONFIGURATION
from mock import patch
from mock.mock import MagicMock
# Save
#
STORAGE_ID = 42
WORKFLOW_ID = 2204

@patch('os.makedirs')
@patch('pandas.DataFrame.to_parquet')
@patch('juicer.service.limonero_service.register_datasource')
@patch('juicer.service.limonero_service.get_storage_info')
def test_save_success(storage: MagicMock, register: MagicMock,
                      parquet: MagicMock, mkdirs: MagicMock):
    slice_size = 10
    df = ['df', util.iris(['sepallength'], slice_size)]

    arguments = {
        'parameters': {
            'configuration': CONFIGURATION,
            'name': 'iris_saved.parquet',
            'format': 'PARQUET',
            'path': '/',
            'user': {'id': 1212235, 'name': 'Tester', 'login': 'tester'},
            'task_id': '8347203473666',
            'storage': STORAGE_ID,
            'workflow_id': WORKFLOW_ID
        },
        'named_inputs': {
            'input data': df[0],
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    storage_result = {
        'id': STORAGE_ID,
        'type': 'HDFS',
        'url': 'file:///tmp/'
    }
    storage.side_effect = lambda *_: storage_result

    instance = SaveOperation(**arguments)
    util.execute(util.get_complete_code(instance),
                          dict([df]))
    storage.assert_called_once_with(
        CONFIGURATION['juicer']['services']['limonero']['url'],
        CONFIGURATION['juicer']['services']['limonero']['auth_token'],
        STORAGE_ID)

    api_payload = {
        'name': 'iris_saved.parquet', 'is_first_line_header': False, 
        'enabled': 1, 'is_public': 0, 'format': 'PARQUET', 'storage_id': 42, 
        'description': f'Data source generated by workflow {WORKFLOW_ID}', 
        'user_id': '1212235', 'user_login': 'tester', 'user_name': 'Tester', 
        'workflow_id': f'{WORKFLOW_ID}', 'task_id': '8347203473666', 
        'url': 'file:///tmp/limonero/user_data/1212235/iris_saved.parquet', 
        'attributes': [
            {'enumeration': 0, 'feature': 0, 'label': 0, 'name': 'sepallength', 
                'type': 'DOUBLE', 'nullable': True, 'metadata': None, 
                'precision': None, 'scale': None}]
    }
    register.assert_called_once_with(
        CONFIGURATION['juicer']['services']['limonero']['url'],
        api_payload,
        CONFIGURATION['juicer']['services']['limonero']['auth_token'],
        'overwrite')
    parquet.assert_called_once_with(
        '/tmp/limonero/user_data/1212235/iris_saved.parquet', engine='pyarrow')
    mkdirs.assert_called_once_with('/tmp/limonero/user_data/1212235', 
        exist_ok=True)
    # assert result['out'].equals(util.iris(size=slice_size))
