# -*- coding: utf-8 -*-
from __future__ import absolute_import

from juicer.util import get_tasks_sorted_topologically, group


def test_get_tasks_sorted_topologically_success():
    workflow = {
        "tasks": [
            {"id": "1"},
            {"id": "2"},
            {"id": "3"}
        ],
        "flows": [
            {
                "source_id": "1",
                "target_id": "2"
            }
        ],
    }
    tasks = get_tasks_sorted_topologically(workflow)
    assert tasks[0]['id'] in ["1", "3"]
    assert tasks[1]['id'] in ["1", "3"]
    assert tasks[2]['id'] == "2"

    ####

    workflow = {
        "tasks": [
            {"id": "1"},
            {"id": "2"},
            {"id": "3"}
        ],
        "flows": [],
    }
    tasks = get_tasks_sorted_topologically(workflow)
    assert tasks[0]['id'] in ["1", "2", "3"]
    assert tasks[1]['id'] in ["1", "2", "3"]
    assert tasks[2]['id'] in ["1", "2", "3"]

    # 1=> 2 => 3
    workflow = {
        "tasks": [
            {"id": "1"},
            {"id": "2"},
            {"id": "3"}
        ],
        "flows": [
            {"source_id": "2", "target_id": "1"},
            {"source_id": "3", "target_id": "2"},
        ],
    }
    tasks = get_tasks_sorted_topologically(workflow)
    assert tasks[0]['id'] in ["3"]
    assert tasks[1]['id'] in ["2"]
    assert tasks[2]['id'] in ["1"]

    # 1=> 2 => 3, 4, 5
    workflow = {
        "tasks": [
            {"id": "1"},
            {"id": "2"},
            {"id": "3"},
            {"id": "4"},
            {"id": "5"},
        ],
        "flows": [
            {"source_id": "2", "target_id": "1"},
            {"source_id": "3", "target_id": "2"},
        ],
    }
    tasks = get_tasks_sorted_topologically(workflow)
    assert tasks[0]['id'] in ["3", "4", "5"]
    assert tasks[1]['id'] in ["3", "4", "5"]
    assert tasks[2]['id'] in ["3", "4", "5"]
    assert tasks[3]['id'] in ["2"]
    assert tasks[4]['id'] in ["1"]

    # 1=> 2 => 4 } => 6
    #  => 3 => 5    }
    workflow = {
        "tasks": [
            {"id": "1"},
            {"id": "2"},
            {"id": "3"},
            {"id": "4"},
            {"id": "5"},
            {"id": "6"},
        ],
        "flows": [
            {"source_id": "4", "target_id": "6"},
            {"source_id": "5", "target_id": "6"},
            {"source_id": "3", "target_id": "5"},
            {"source_id": "2", "target_id": "4"},
            {"source_id": "1", "target_id": "3"},
            {"source_id": "1", "target_id": "2"},
        ],
    }
    tasks = get_tasks_sorted_topologically(workflow)
    assert tasks[0]['id'] in ["1"]
    assert tasks[1]['id'] in ["2", "3"]
    assert tasks[2]['id'] in ["2", "3"]
    assert tasks[3]['id'] in ["4", "5"]
    assert tasks[4]['id'] in ["4", "5"]
    assert tasks[5]['id'] in ["6"]


def test_group_success():
    assert sorted(group([0, 3, 4, 10, 2, 3], 2)) == sorted(
        [(0, 3), (4, 10), (2, 3)])
    assert sorted(group(list(range(10)), 3)) == sorted(
        [(0, 1, 2), (3, 4, 5), (6, 7, 8)])
