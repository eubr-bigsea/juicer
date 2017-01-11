# -*- coding: utf-8 -*-
import json

import sys

import signal

import errno

import mock
from juicer.runner.juicer_server import JuicerServer
from juicer.runner.control import StateControlRedis
from mockredis.client import mock_strict_redis_client


def test_runner_read_start_queue_success():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    job_id = 1
    workflow_id = 1000
    workflow = {"job_id": job_id, 'workflow_id': workflow_id}

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = server = JuicerServer(config, 'faked_minions.py',
                                           config_file_path='config.yaml')
            mocked_redis_conn = mocked_redis()
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))

            # Start of testing
            server.read_start_queue(mocked_redis_conn)

            assert state_control.get_minion_status(
                job_id) == JuicerServer.STARTED

            assert mocked_popen.call_args_list[0][0][0] == [
                'nohup', sys.executable, 'faked_minions.py', '-j', job_id, '-c',
                'config.yaml']
            assert mocked_popen.called

            # Was command removed from the queue?
            assert state_control.pop_start_queue(False) is None

            assert json.loads(state_control.pop_job_queue(job_id)) == workflow

            assert state_control.get_workflow_status(
                workflow_id) == JuicerServer.STARTED
            assert json.loads(state_control.pop_job_output_queue(job_id)) == {
                'code': 0, 'message': 'Workflow will start soon'}


def test_runner_read_start_queue_already_started_failure():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    job_id = 1
    workflow_id = 1000
    workflow = {"job_id": job_id, 'workflow_id': workflow_id}

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = server = JuicerServer(config, 'faked_minions.py')
            mocked_redis_conn = mocked_redis()
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))

            # This workflow is being processed, should not start it again
            state_control.set_workflow_status(workflow_id, JuicerServer.STARTED)

            # Start of testing
            server.read_start_queue(mocked_redis_conn)

            assert state_control.get_minion_status(job_id) is None
            assert not mocked_popen.called
            # Was command removed from the queue?
            assert state_control.pop_start_queue(False) is None

            assert state_control.get_job_queue_size(workflow_id) == 0

            assert state_control.get_workflow_status(
                workflow_id) == JuicerServer.STARTED

            x = state_control.pop_job_output_queue(job_id, False)
            assert json.loads(
                x) == {
                       "message": "Workflow is already started", "code": 1000}


def test_runner_read_start_queue_minion_already_running_success():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    job_id = 1
    workflow_id = 1000
    workflow = {"job_id": job_id, 'workflow_id': workflow_id}

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = JuicerServer(config, 'faked_minions.py')
            mocked_redis_conn = mocked_redis()
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))
            state_control.set_minion_status(job_id, JuicerServer.STARTED)

            # Start of testing
            server.read_start_queue(mocked_redis_conn)

            assert state_control.get_minion_status(
                job_id) == JuicerServer.STARTED

            assert not mocked_popen.called
            # Was command removed from the queue?
            assert mocked_redis_conn.lpop('start') is None
            assert json.loads(state_control.pop_job_queue(job_id)) == workflow

            assert state_control.get_workflow_status(
                workflow_id) == JuicerServer.STARTED
            assert json.loads(state_control.pop_job_output_queue(job_id)) == {
                'code': 0, 'message': 'Workflow will start soon'}


def test_runner_read_start_queue_missing_details_failure():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    job_id = 1
    workflow_id = 1000
    # incorrect key, should raise exception
    workflow = {"xjob_id": job_id, 'workflow_id': workflow_id}

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            server = JuicerServer(config, 'faked_minions.py')
            mocked_redis_conn = mocked_redis()
            # Publishes a message to process data
            state_control = StateControlRedis(mocked_redis_conn)

            # Publishes a message to process data
            state_control.push_start_queue(json.dumps(workflow))
            state_control.set_minion_status(job_id, JuicerServer.STARTED)

            # Start of testing
            server.read_start_queue(mocked_redis_conn)

            assert state_control.get_minion_status(
                job_id) == JuicerServer.STARTED

            assert not mocked_popen.called
            # Was command removed from the queue?
            assert state_control.pop_start_queue(block=False) is None
            assert state_control.pop_job_queue(job_id, block=False) is None
            assert state_control.pop_job_output_queue(job_id,
                                                      block=False) is None

            assert mocked_redis_conn.hget(workflow_id, 'status') is None


def test_runner_master_queue_client_shutdown_success():
    config = {
        'juicer': {
            'servers': {
                'redis_url': "nonexisting.mock"
            }
        }
    }
    job_id = 1
    ticket = {"job_id": job_id, 'reason': JuicerServer.HELP_UNHANDLED_EXCEPTION}

    with mock.patch('redis.StrictRedis',
                    mock_strict_redis_client) as mocked_redis:
        with mock.patch('subprocess.Popen') as mocked_popen:
            with mock.patch('os.kill') as mocked_kill:
                server = JuicerServer(config, 'faked_minions.py')
                mocked_redis_conn = mocked_redis()
                # Publishes a message to process data
                state_control = StateControlRedis(mocked_redis_conn)

                # Configure minion
                status = {'job_id': job_id, 'pid': 9999}
                state_control.set_minion_status(job_id, json.dumps(status))
                error = OSError()
                error.errno = errno.ESRCH
                mocked_kill.side_effect = error
                # Publishes a message to master queue
                state_control.push_master_queue(json.dumps(ticket))

                # Start of testing
                server.read_minion_support_queue(mocked_redis_conn)

                assert state_control.get_minion_status(job_id) == 'STARTED'

                assert mocked_popen.called

                mocked_kill.assert_called_once_with(status['pid'],
                                                    signal.SIGKILL)

