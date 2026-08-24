# -*- coding: utf-8 -*-
"""
S6 reliability fix: the Jinja/gettext Environment and the audit-job Redis
Queue used to be rebuilt from scratch on every generate_code() call. Both are
fully determined by their input (template_dir / redis_url), so they are now
memoized and reused for the life of the process.
"""
from juicer.transpiler import _build_template_env, _get_audit_queue


def test_build_template_env_is_cached_per_template_dir():
    env1 = _build_template_env('juicer/spark/templates')
    env2 = _build_template_env('juicer/spark/templates')
    assert env1 is env2

    env3 = _build_template_env('juicer/scikit_learn/templates')
    assert env3 is not env1


def test_get_audit_queue_is_cached_per_redis_url():
    q1 = _get_audit_queue('redis://localhost:6379/0')
    q2 = _get_audit_queue('redis://localhost:6379/0')
    assert q1 is q2

    q3 = _get_audit_queue('redis://otherhost:6379/0')
    assert q3 is not q1
