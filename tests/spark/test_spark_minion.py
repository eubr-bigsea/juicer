# coding=utf-8
from __future__ import print_function

import json
from textwrap import dedent

from datetime import datetime

import mock
import os
from dummy_spark import SparkConf, SparkContext
from juicer.runner.control import StateControlRedis
from juicer.spark.spark_minion import SparkMinion
from mockredis import mock_strict_redis_client

config = {
    'juicer': {
        'servers': {
            'redis_url': 'redis://invalid:2923'
        }
    }
}


class DataFrame:
    def __init__(self, rdd):
        self.rdd = rdd

    def take(self, total):
        return self


# noinspection PyProtectedMember
def test_minion_ping_success():
    job_id = 897987

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        redis_conn = mocked_redis()
        minion = SparkMinion(redis_conn=redis_conn, job_id=job_id,
                             config=config)
        minion._perform_ping()

        state_control = StateControlRedis(redis_conn)

        assert json.loads(state_control.get_minion_status(job_id)) == {
            'status': 'READY', 'pid': os.getpid()}
        assert state_control.get_job_output_queue_size(job_id) == 0


# noinspection PyProtectedMember
def test_minion_generate_output_success():
    job_id = 897987

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        redis_conn = mocked_redis()
        minion = SparkMinion(redis_conn=redis_conn, job_id=job_id,
                             config=config)

        state_control = StateControlRedis(redis_conn)

        msgs = ["Message being sent \n{}", {'msg': 'Dictionary being sent'}]
        for msg in msgs:
            minion._generate_output(msg)
            result = json.loads(
                state_control.pop_job_output_queue(job_id, False))
            assert result.keys() == ['date', 'status', 'message', 'job_id']
            assert result['job_id'] == job_id
            assert result['message'] == msg
        assert state_control.get_job_output_queue_size(job_id) == 0


# noinspection PyProtectedMember
def test_minion_perform_execute_success():
    job_id = 897447
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    # noinspection PyUnusedLocal
    def side_effect(w, g, c, out):
        print(dedent("""
            def main():
                return {
                    "xyz-647": ("df", 27.27)
                }"""), file=out)

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = side_effect

            redis_conn = mocked_redis()
            minion = SparkMinion(redis_conn=redis_conn, job_id=job_id,
                                 config=config)

            # Configure mocked redis
            state_control = StateControlRedis(redis_conn)
            with open(os.path.join(os.path.dirname(__file__),
                                   'fixtures/simple_workflow.json')) as f:
                data = json.loads(f.read())

            state_control.push_job_queue(job_id, json.dumps({'workflow': data}))

            minion._perform_execute()

            assert minion.state == {"xyz-647": ("df", 27.27)}, 'Invalid state'

            assert state_control.get_job_output_queue_size(
                job_id) == 1, 'Wrong number of output messages'


# noinspection PyProtectedMember
def test_minion_perform_execute_reload_code_success():
    job_id = 667788
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    # noinspection PyUnusedLocal
    def side_effect1(w, g, c, out):
        print(dedent("""
            def main():
                return 'version 1.0' """), file=out)

    def side_effect2(w, g, c, out):
        print(dedent("""
            def main():
                return 'version 2.1' """), file=out)

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = side_effect1

            redis_conn = mocked_redis()
            minion = SparkMinion(redis_conn=redis_conn, job_id=job_id,
                                 config=config)

            # Configure mocked redis
            state_control = StateControlRedis(redis_conn)
            with open(os.path.join(os.path.dirname(__file__),
                                   'fixtures/simple_workflow.json')) as f:
                data = json.loads(f.read())

            state_control.push_job_queue(job_id, json.dumps({'workflow': data}))

            minion._perform_execute()
            assert minion.state == 'version 1.0', 'Invalid state'

            # Executes the same workflow, but code should be different
            state_control.push_job_queue(job_id, json.dumps({'workflow': data}))
            assert state_control.get_job_output_queue_size(
                job_id) == 1, 'Wrong number of output messages'

            minion.transpiler.transpile = side_effect2
            minion._perform_execute()

            assert minion.state == 'version 2.1', 'Invalid state'

            assert state_control.get_job_output_queue_size(
                job_id) == 2, 'Wrong number of output messages'


# noinspection PyProtectedMember
def test_minion_generate_invalid_code_failure():
    job_id = 666
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'

    # noinspection PyUnusedLocal
    def side_effect1(w, g, c, out):
        print(dedent("""
            def main():
                return Invalid Code """), file=out)

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = side_effect1
            redis_conn = mocked_redis()
            minion = SparkMinion(redis_conn=redis_conn, job_id=job_id,
                                 config=config)
            # Configure mocked redis
            with open(os.path.join(os.path.dirname(__file__),
                                   'fixtures/simple_workflow.json')) as f:
                data = json.loads(f.read())
            state_control = StateControlRedis(redis_conn)
            state_control.push_job_queue(job_id, json.dumps({'workflow': data}))
            minion._perform_execute()

            assert state_control.get_job_output_queue_size(
                job_id) == 2, 'Wrong number of output messages'
            # Discards
            state_control.pop_job_output_queue(job_id)

            msg = json.loads(state_control.pop_job_output_queue(job_id))
            assert msg['status'] == 'ERROR'
            assert msg['message'][:19] == 'Invalid Python code'


# noinspection PyProtectedMember
def test_minion_perform_deliver_success():
    job_id = 1000
    out_queue = 'queue_2000'
    sconf = SparkConf()
    sc = SparkContext(master='', conf=sconf)
    records = [
        ['Brazil', 'UFMG', 223, datetime.now()],
        ['Spain', 'UPV', 52, datetime.now()],
        ['Italy', 'POLIMI', 921, datetime.now()],
    ]
    rdd = sc.parallelize(records)

    df0 = DataFrame(rdd=rdd)
    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        redis_conn = mocked_redis()
        state_control = StateControlRedis(redis_conn)

        data = {
            'task_id': '033f-284ab-28987e',
            'port': '0',
            'output': out_queue
        }
        state_control.push_job_delivery_queue(job_id, json.dumps(data))
        minion = SparkMinion(redis_conn=redis_conn, job_id=job_id,
                             config=config)
        minion.state = {
            data['task_id']: (df0, 35.92)
        }
        minion._perform_deliver()

        msg = json.loads(state_control.pop_job_output_queue(job_id, False))
        assert msg['status'] == 'SUCCESS', 'Invalid status'
        assert msg['code'] == SparkMinion.MNN002[0], 'Invalid code'

        # CSV data
        csv_records = '\n'.join(map(minion._convert_to_csv, records))

        result = state_control.pop_queue(out_queue, False)
        assert result == csv_records, 'Wrong CSV generated'


# noinspection PyProtectedMember
def test_minion_perform_deliver_missing_state_process_job_with_success():
    job_id = 1000
    out_queue = 'queue_2000'
    task_id = '033f-284ab-28987e'
    function_name = 'juicer.spark.transpiler.SparkTranspiler.transpile'
    records = [
        ['Speed', 'UFMG', 1223],
        ['Dev', 'UPV', 524],
        ['QoS', 'POLIMI'],
    ]

    # noinspection PyUnusedLocal
    def side_effect1(w, g, c, out):
        print(dedent("""
            from dummy_spark import SparkConf, SparkContext
            import datetime
            sconf = SparkConf()
            sc = SparkContext(master='', conf=sconf)
            records = {records}
            rdd = sc.parallelize(records)
            class DataFrame:
                def __init__(self, rdd):
                    self.rdd = rdd

                def take(self, total):
                    return self
            def main():
                return {{
                    '{task_id}': (DataFrame(rdd), 20.0)
                }}
            """.format(records=json.dumps(records), task_id=task_id)), file=out)

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch(function_name) as mocked_transpile:
            # Setup for mocked_transpile
            mocked_transpile.side_effect = side_effect1
            redis_conn = mocked_redis()
            state_control = StateControlRedis(redis_conn)

            data = {
                'task_id': task_id,
                'port': '0',
                'output': out_queue,
                'workflow': '{"workflow": {"tasks": [], "flows":[]}}'
            }

            state_control.push_job_delivery_queue(job_id, json.dumps(data))
            minion = SparkMinion(redis_conn=redis_conn, job_id=job_id,
                                 config=config)
            minion.state = {
            }
            minion._perform_deliver()

            msg = json.loads(state_control.pop_job_output_queue(job_id, False))
            assert msg['status'] == 'WARNING', 'Invalid status'
            assert msg['code'] == SparkMinion.MNN003[0], 'Invalid code'

            # CSV data
            csv_records = '\n'.join(map(minion._convert_to_csv, records))

            result = state_control.pop_queue(out_queue, False)
            assert result == csv_records, 'Wrong CSV generated'


def test_minion_perform_deliver_missing_state_process_job_with_failure():
    assert False, 'To be implemented'


def test_minion_perform_deliver_invalid_workflow_with_failure():
    assert False, 'To be implemented'


def test_minion_perform_deliver_unsupported_output_failure():
    assert False, 'To be implemented'


def test_minion_perform_deliver_invalid_port_failure():
    assert False, 'To be implemented'
