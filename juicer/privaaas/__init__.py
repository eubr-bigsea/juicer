# -*- coding: utf-8 -*-
"""PRIVAaaS - Privacy as a Service
EUBra-BIGSEA project.

This module implements methods to guarantee data privacy during the execution
of Lemonade's workflows.

"""
import hashlib
import json
import sys
from textwrap import dedent

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

ANONYMIZATION_TECHNIQUES = {
    'NO_TECHNIQUE': 0,
    'GENERALIZATION': 1,
    'MASK': 2,
    'ENCRYPTION': 3,
    'SUPPRESSION': 4
}


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
                value_unit = sys.maxint
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


def generalization(details):
    details = json.loads(details)

    def _apply(value):
        if type(value) is int and details.startswith('('):
            len_truncation = int(details[1:-1])
            result = _truncate_number(len_truncation, value)
        elif type(value) in [unicode, str] and details.startswith('('):
            len_truncation = int(details[1:-1])
            result = _truncate_number(len_truncation, value)
        elif (type(value) is int or type(
                value) is float) and details.startswith(
                '['):
            result = _number_range_to_string(value, details)
        elif type(value) in [unicode, str] and details.startswith('{'):
            result = _string_range_to_string(value, details)
        else:
            raise ValueError(_('Invalid hierarchy for generalization'))
        return result

    return udf(_apply)  # FIXME: How handle correct type?


def encryption(details):
    """
    Returns a Spark compatible UDF (user defined function) to encrypt data in
    a data frame column.
     :param details Parameters for encryption, basically, the name of algorithm
     to be used;
    """
    details = json.loads(details)
    algorithm = details.get('algorithm', 'sha1')

    if algorithm == 'md5':
        h = hashlib.md5().update
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

    return udf(_apply, StringType())


def masking_gen(attribute_name, details):
    """
    Apply masking to a RDD of rows. Rows are first grouped by key in order to
    have rows with same value for the column available at same time (if the
    value is the same, the mask will be the same).
    @FIXME: Define a good size for partitions / groups (for instance use part
    of string or range of numbers, but it depends on the data type).
    """
    print '-' * 20
    print details
    print '-' * 20
    details = json.loads(details)

    def masking(group):
        from faker import Factory
        faker_obj = Factory.create(details.get('lang', 'en_GB'))

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
            # Privacy policy: attribute suppression')
            {out} = {out}.drop(*{cols})""".format(
            out=self.output, cols=json.dumps([g['name'] for g in group])))

    def _exec_as_udf(self, group, action):
        code = ["# Privacy policy: attribute ".format(action)]

        template = dedent("""
            privaas_udf = privaaas.{f}({details})
            {out} = {out}.withColumn(colName='{name}',
                col=privaas_udf('{name}'))""").strip()

        for g in group:
            code.append(template.format(out=self.output, name=g['name'],
                                        details=repr(g['details']),
                                        f="encryption"))
        return '\n'.join(code)

    def encryption(self, group):
        self._exec_as_udf(group, 'encryption')

    def generalization(self, group):
        self._exec_as_udf(group, 'generalization')

    def mask(self, group):
        code = ['', '# PRIVAaS Privacy policy: attribute mask']

        template = dedent("""
            details = {details}
            col_names = [s.name for s in {out}.schema]
            col_type = {out}.schema['{name}'].dataType
            if isinstance(col_type, types.StringType):
                key_gen = lambda x: x['{name}'][0] if x['{name}'] else ""
            elif isinstance(col_type, types.IntegralType):
                key_gen = lambda x: x['{name}'] if x['{name}'] % 1 else 0
            elif isinstance(col_type, types.FractionalType):
                key_gen = lambda x: x['{name}'] % 1 if x['{name}'] else 0
            # elif isinstance(col_type, types.DateType):
            else:
                raise ValueError('{not_supported}'.format(col_type))

            {out} = {out}.rdd.keyBy(key_gen).groupByKey().flatMap(
                    privaaas.{f}('{name}', details)).map(
                        lambda l: Row(**dict(l))).toDF()""").strip()
        for g in group:
            code.append(template.format(
                out=self.output, name=g['name'], f="masking_gen",
                not_supported=_('Data type is not supported for masking: {}'),
                details=repr(g['details'])))
        code.append('')
        return '\n'.join(code)
