# -*- coding: utf-8 -*-

import functools
from collections import defaultdict
from itertools import takewhile, count


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def sort_topologically(graph):
    """
    Topological sorting. See http://stackoverflow.com/a/15039202/1646932.
    """
    levels_by_name = {}
    names_by_level = defaultdict(set)

    def walk_depth_first(level_name):
        if level_name in levels_by_name:
            return levels_by_name[level_name]
        children = graph.get(level_name, None)
        level = 0 if not children else \
            (1 + max(walk_depth_first(lname) for lname in children))
        levels_by_name[level_name] = level
        names_by_level[level].add(level_name)
        return level

    for name in graph:
        walk_depth_first(name)

    return list(takewhile(lambda x: x is not None,
                          (names_by_level.get(i, None) for i in count())))


def get_tasks_sorted_topologically(workflow):
    """
    Sorts a Lemonade workflow topologically (respects dependencies between
    tasks).
    """
    graph = {}
    all_tasks = {}
    for task in workflow['tasks']:
        graph[task['id']] = []
        all_tasks[task['id']] = task
    for flow in workflow['flows']:
        graph[flow['target_id']].append(flow['source_id'])

    dependency = sort_topologically(graph)
    return [all_tasks[item] for sublist in dependency for item in sublist]


def group(lst, n):
    """group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]

    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.
    > group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    return list(zip(*[lst[i::n] for i in range(n)]))


def get_emitter(emit_event, operation_id, task_id, title=''):
    return functools.partial(
        emit_event, name='update task',
        status='RUNNING', type='TEXT',
        identifier=task_id,
        operation={'id': operation_id}, operation_id=operation_id,
        task={'id': task_id},
        title=title)
