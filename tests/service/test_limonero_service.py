# coding=utf-8
from __future__ import absolute_import
import json

import pytest
from juicer.service import limonero_service
from mock import patch
from . import fake_req


@patch('requests.get')
def test_get_storage_info_success(mocked_get):
    storage_id = 700
    text = {
        'id': storage_id,
        'name': 'Storage HDFS',
        'url': 'hdfs://test.com'
    }
    mocked_get.side_effect = fake_req(200, json.dumps(text))()
    url = 'http://limonero'
    token = '00000'

    resp = limonero_service.get_storage_info(url, token, storage_id)
    for k, v in resp.items():
        assert v == text[k]

    mocked_get.assert_called_with(
        'http://limonero/storages/{}'.format(storage_id),
        headers={'X-Auth-Token': '00000'})


@patch('requests.get')
def test_get_storage_info_failure(mocked_get):
    storage_id = 700
    text = {
        'id': storage_id,
        'name': 'Storage HDFS',
        'url': 'hdfs://test.com'
    }
    mocked_get.side_effect = fake_req(201, json.dumps(text))()
    url = 'http://limonero'
    token = '00000'
    with pytest.raises(ValueError):
        resp = limonero_service.get_storage_info(url, token, storage_id)
        mocked_get.assert_called_with(
            'http://limonero/storages/{}'.format(storage_id),
            headers={'X-Auth-Token': '00000'})
        for k, v in resp.items():
            assert v == text[k]


@patch('requests.get')
def test_get_data_source_info_success(mocked_get):
    data_source_id = 700
    text = {
        'id': data_source_id,
        'name': 'Data source for testing',
        'url': 'hdfs://test.com/testing.csv'
    }
    mocked_get.side_effect = fake_req(200, json.dumps(text))()
    url = 'http://limonero'
    token = '00000'

    resp = limonero_service.get_data_source_info(url, token, data_source_id)
    for k, v in resp.items():
        assert v == text[k]

    mocked_get.assert_called_with(
        'http://limonero/datasources/{}'.format(data_source_id),
        headers={'X-Auth-Token': '00000'})


@patch('requests.get')
def test_get_all_data_sources_success(mocked_get):
    data_source_id = 700
    text = {
        'id': data_source_id,
        'name': 'Data source for testing',
        'url': 'hdfs://test.com/testing.csv'
    }
    mocked_get.side_effect = fake_req(200, json.dumps(text))()
    url = 'http://limonero/'
    token = '00000'

    resp = limonero_service.get_data_source_info(url, token, '')
    for k, v in resp.items():
        assert v == text[k]

    mocked_get.assert_called_with(
        'http://limonero/datasources/',
        headers={'X-Auth-Token': '00000'})


@patch('requests.get')
def test_get_data_source_info_failure(mocked_get):
    data_source_id = 700
    text = {
        'id': data_source_id,
        'name': 'Data source for testing',
        'url': 'hdfs://test.com/testing.csv'
    }
    mocked_get.side_effect = fake_req(201, json.dumps(text))()
    url = 'http://limonero/datasources'
    token = '00000'
    with pytest.raises(ValueError):
        resp = limonero_service.get_data_source_info(url, token, data_source_id)

        mocked_get.assert_called_with(
            'http://limonero/datasources/{}'.format(data_source_id),
            headers={'X-Auth-Token': '00000'})
        for k, v in resp.items():
            assert v == text[k]
