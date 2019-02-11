# coding=utf-8
import codecs
import imp
import io

import mock

from juicer.spark.data_operation import DataReaderOperation, SaveOperation
from juicer.workflow.workflow import Workflow


class EmitStore(object):
    items = []

    def emit_list(self, *a, **kwargs):
        if kwargs.get('type') == 'HTML':
            del kwargs['message']
        self.items.append(kwargs)


def xtest_data_reader_simple_csv_success(
        spark_session, spark_transpiler, juicer_config_for_spark,
        spark_operations, iris_data, iris_workflow):
    get_ops = 'juicer.workflow.workflow.Workflow._get_operations'
    limonero_op = 'juicer.spark.data_operation.DataReaderOperation.' \
                  '_get_data_source_info'

    with mock.patch(get_ops) as mocked_operations:
        mocked_operations.side_effect = lambda: spark_operations
        with mock.patch(limonero_op) as get_ds:
            get_ds.side_effect = lambda: iris_data

            loader = Workflow(iris_workflow, juicer_config_for_spark)
            code_buffer = io.StringIO()
            spark_transpiler.transpile(
                loader.workflow, loader.graph, {}, out=code_buffer, job_id=1)

            code_buffer.seek(0)
            test_module = imp.new_module('test_juicer')
            exec (code_buffer.read().encode('utf8'), test_module.__dict__)

            es = EmitStore()
            test_module.main(spark_session, {}, es.emit_list)

            assert len(es.items) == 3
            assert ["RUNNING", "COMPLETED", "COMPLETED"] == [item['status'] for
                                                             item in es.items]
            assert es.items[1]['type'] == 'HTML'


def xtest_data_writer_csv_success(
        spark_session,
        spark_transpiler, juicer_config_for_spark,
        spark_operations, iris_workflow, iris_data):
    wf = {}
    wf.update(iris_workflow)
    wf['flows'] = [
        {'source_id': '001', 'source_port_name': 'output data',
         'target_id': '002', 'target_port_name': 'input data'}
    ]
    wf['tasks'].append(
        {
            'id': '002',
            'operation': {
                'id': 1, 'slug': SaveOperation.SLUG,
            },
            'forms': {
                'display_sample': {'category': 'EXECUTION', 'value': 1},
                SaveOperation.NAME_PARAM:
                    {'category': 'EXECUTION', 'value': 'new_iris'},
                SaveOperation.FORMAT_PARAM:
                    {'category': 'EXECUTION',
                     'value': SaveOperation.FORMAT_CSV},
                SaveOperation.STORAGE_ID_PARAM:
                    {'category': 'EXECUTION', 'value': 1},
                SaveOperation.PATH_PARAM:
                    {'category': 'EXECUTION', 'value': '/data/'},
            }
        }
    )
    get_ops = 'juicer.workflow.workflow.Workflow._get_operations'
    limonero_op = 'juicer.spark.data_operation.DataReaderOperation.' \
                  '_get_data_source_info'

    with mock.patch(get_ops) as mocked_operations:
        mocked_operations.side_effect = lambda: spark_operations
        with mock.patch(limonero_op) as get_ds:
            get_ds.side_effect = lambda: iris_data

            loader = Workflow(wf, juicer_config_for_spark)
            code_buffer = io.StringIO()
            spark_transpiler.transpile(
                loader.workflow, loader.graph, {}, out=code_buffer, job_id=1)

            code_buffer.seek(0)
            test_module = imp.new_module('test_juicer')
            exec (code_buffer.read().encode('utf8'), test_module.__dict__)

            es = EmitStore()
            test_module.main(spark_session, {}, es.emit_list)

            code_buffer.seek(0)
            with codecs.open('/tmp/juicer_app_1_1_1.py', 'w', 'utf8') as f:
                f.write(code_buffer.read())
