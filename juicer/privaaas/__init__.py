# -*- coding: utf-8 -*-
"""PRIVAaaS - Privacy as a Service
EUBra-BIGSEA project.

This module implements methods to guarantee data privacy during the execution
of Lemonade's workflows.

"""


import hashlib
import json
import math
import numbers
import random
import sys
from textwrap import dedent

import datetime

from six import text_type

ANONYMIZATION_TECHNIQUES = {
    'NO_TECHNIQUE': 0,
    'GENERALIZATION': 1,
    'MASK': 2,
    'ENCRYPTION': 3,
    'SUPPRESSION': 4
}


def generate_group_by_key_function(df, col_name):
    from pyspark.sql import types
    col_type = df.schema[col_name].dataType

    # Spark 2.2 has problems with functools.partial:
    # AttributeError: 'functools.partial' object has no attribute '__module__'
    # See https://issues.apache.org/jira/browse/SPARK-21432
    def key_gen_str(row):
        return row[col_name][0] if row[col_name] else ""

    def key_gen_int(row):
        return row[col_name] % 1 if row[col_name] else 0

    def key_gen_float(row):
        return row[col_name] % 1 if row[col_name] else 0

    if isinstance(col_type, bytes):
        key_gen = key_gen_str
    elif isinstance(col_type, types.IntegralType):
        key_gen = key_gen_int
    elif isinstance(col_type, types.FractionalType):
        key_gen = key_gen_float
    # elif isinstance(col_type, types.DateType):
    else:
        raise ValueError(
            'Data type is not supported for masking: {}'.format(col_type))
    return key_gen


def _truncate_number(value, len_truncation):
    """ Truncates a number until len_truncation.
     @FIXME: Review implementation
    """
    return int(str(value)[:len_truncation])


def _truncate_string(value, len_truncation):
    len_value = len(value)
    complete_value = len_value - len_truncation
    truncate_value = str(value)
    return truncate_value[:len_truncation] + '*' * complete_value


def _number_range_to_string(value, details):
    detail_value = details[1:-1]
    parts = detail_value.split(';')
    list_of_parts = list(parts)

    for ranges in list_of_parts:
        range_value = ranges.split('=')
        values = range_value[0].split('-')
        list_of_integers = []
        for k in values:
            if k != 'x':
                value_unit = int(k)
            else:
                value_unit = sys.maxsize
            list_of_integers.append(value_unit)
        min_value = list_of_integers[0]
        max_value = list_of_integers[1]
        if min_value <= value <= max_value:
            return range_value[1]


def _string_range_to_string(value, details):
    detail_value = details[1:-1].replace(' ', '')
    parts = detail_value.split(';')
    list_of_parts = list(parts)
    for ranges in list_of_parts:
        range_value = ranges.split('=')
        values = range_value[0]
        values_str = values[1:-1]
        values_str_unit = values_str.split(',')
        for item in values_str_unit:
            if item == value:
                return range_value[1]


def _number_to_bucket(value, bucket_size):
    result = None
    if isinstance(value, numbers.Integral):
        start = int(math.ceil(value / bucket_size) * bucket_size - 1)
        end = int(math.floor(value / bucket_size) * bucket_size)
        result = [start, end]
    elif isinstance(value, numbers.Real):
        start = float(math.ceil(value / bucket_size) * bucket_size - 1)
        end = float(math.floor(value / bucket_size) * bucket_size)
        result = [start, end]
    elif isinstance(value, (str, text_type)):
        # @FIXME Implement
        result = ''
    elif isinstance(value, (datetime.datetime, datetime.date)):
        pass
    return result


def _substring(value, _from, to, size, replacement):
    if value is None:
        result = replacement * size
    else:
        if size == -1:
            size = len(value)
        if isinstance(value, text_type):
            replacement = replacement
        else:
            replacement = str(replacement)
        result = value[_from:to].ljust(size, replacement)
    return result


def _hierarchy(value, values, default):
    return values.get(value, default)


def generalization(details):
    gen_type = details.get('type')

    # Spark 2.2 has problems with functools.partial:
    # AttributeError: 'functools.partial' object has no attribute '__module__'
    # See https://issues.apache.org/jira/browse/SPARK-21432

    result = None
    if gen_type == 'range':

        bucket_size = details.get('range_args', {}).get('bucket_size') or 5

        def _inner_number_to_bucket(v):
            return _number_to_bucket(v, bucket_size=bucket_size)

        result = _inner_number_to_bucket
    elif gen_type == 'substring':
        args = details.get('substring_args', {})
        _from = args.get('from') or 0
        to = args.get('to') or 1
        size = args.get('final_size') or 10
        replacement = args.get('replacement') or '*'

        def _inner_substring(value):
            return _substring(value, _from, to, size, replacement)

        result = _inner_substring
        # result = functools.partial(_substring, _from=_from, to=to,
        #                            size=size, replacement=replacement)
    elif gen_type == 'hierarchy':
        args = details.get('hierarchy_args', {})
        values = args.get('values', {})
        default = args.get('default')

        def _inner_hierarchy(value):
            return _hierarchy(value, values, default)

        result = _inner_hierarchy
        # result = functools.partial(_hierarchy, values=values,
        #                            default=default)

    return result


def serializable_md5(v):
    h = hashlib.md5()
    h.update(v)
    return h


def encryption(details):
    """
    Returns a Spark compatible UDF (user defined function) to encrypt data in
    a data frame column.
     :param details Parameters for encryption, basically, the name of algorithm
     to be used;
    """
    algorithm = details.get('algorithm', 'sha1')

    if algorithm == 'md5':
        h = serializable_md5
    elif algorithm == 'sha1':
        h = hashlib.sha1
    elif algorithm == 'sha224':
        h = hashlib.sha224
    elif algorithm == 'sha256':
        h = hashlib.sha256
    elif algorithm == 'sha384':
        h = hashlib.sha384
    elif algorithm == 'sha512':
        h = hashlib.sha512
    else:
        raise ValueError(_('Invalid encryption function {}').format(algorithm))

    def _apply(value):
        return h(value).hexdigest()

    return _apply


def masking_gen(attribute_name, details):
    """
    Apply masking to a RDD of rows. Rows are first grouped by key in order to
    have rows with same value for the column available at same time (if the
    value is the same, the mask will be the same).
    @FIXME: Define a good size for partitions / groups (for instance use part
    of string or range of numbers, but it depends on the data type).
    """

    def masking(group):
        from faker import Factory
        faker_obj = Factory.create(details.get('lang', 'en_GB'))
        faker_obj.seed(random.randint(0, 100000))

        if not hasattr(faker_obj, details.get('label_type', 'name')):
            raise ValueError(_('Invalid masking type: {}').format(
                details.get('label_type')))

        action = getattr(faker_obj, details.get('label_type', 'name'))
        faker_ctx = {}

        result = []
        for row in group[1]:
            as_dict = row.asDict()
            value = as_dict.get(attribute_name)
            if value in faker_ctx:
                new_value = faker_ctx.get(value)
            else:
                new_value = action(**details.get('label_args', {}))
                faker_ctx[value] = new_value

            as_dict[attribute_name] = new_value
            result.append(as_dict)
        return result

    return masking


class PrivacyPreservingDecorator(object):
    def __init__(self, output):
        self.output = output

    def suppression(self, group):
        return dedent("""
            # PRIVAaaS Privacy policy: attribute suppression
            {out} = {out}.drop(*{cols})""".format(
            out=self.output, cols=json.dumps([g['name'] for g in group])))

    def _exec_as_udf(self, group, action):
        code = ["# PRIVAaS Privacy policy: attribute {}".format(action)]
        template = dedent("""
            details = {details}
            anonymization = '{f}'
            if details.get('type') == 'range' and \\
                anonymization == 'generalization':
                attr = next(x for x in {out}.schema if x.name == '{name}')
                return_type = types.ArrayType(attr.dataType)
            else:
                return_type = types.StringType()

            privaaas_udf = functions.udf(privaaas.{f}(details),
                return_type)
            {out} = {out}.withColumn(colName='{name}',
                col=privaaas_udf('{name}'))""").strip()

        for g in group:
            code.append(template.format(out=self.output, name=g['name'],
                                        details=g['details'],
                                        f=action))
        return '\n'.join(code)

    def encryption(self, group):
        return self._exec_as_udf(group, 'encryption')

    def generalization(self, group):
        return self._exec_as_udf(group, 'generalization')

    def mask(self, group):
        code = ['', '# PRIVAaS Privacy policy: attribute mask']

        template = dedent("""
            details = {details}
            key_gen = privaaas.generate_group_by_key_function({out}, '{name}')
            {out} = {out}.rdd.keyBy(key_gen).groupByKey().flatMap(
                    privaaas.{f}('{name}', details)).map(
                        lambda l: Row(**dict(l))).toDF({out}.schema)""").strip()
        for g in group:
            code.append(template.format(
                out=self.output, name=g['name'], f="masking_gen",
                not_supported=_('Data type is not supported for masking: {}'),
                details=g['details']))
        code.append('')
        return '\n'.join(code)
