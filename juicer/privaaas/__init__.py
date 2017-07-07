# -*- coding: utf-8 -*-
"""PRIVAaaS - Privacy as a Service
EUBra-BIGSEA project.

This module implements methods to guarantee data privacy during the execution
of Lemonade's workflows.

"""
import hashlib
import sys

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


def _number_range_to_string(value, detail):
    detail_value = detail[1:-1]
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


def _string_range_to_string(value, detail):
    detail_value = detail[1:-1].replace(' ', '')
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


def generalization(value, detail):
    if type(value) is int and detail.startswith('('):
        len_truncation = int(detail[1:-1])
        result = _truncate_number(len_truncation, value)
    elif type(value) in [unicode, str] and detail.startswith('('):
        len_truncation = int(detail[1:-1])
        result = _truncate_number(len_truncation, value)
    elif (type(value) is int or type(value) is float) and detail.startswith(
            '['):
        result = _number_range_to_string(value, detail)
    elif type(value) in [unicode, str] and detail.startswith('{'):
        result = _string_range_to_string(value, detail)
    else:
        raise ValueError('Invalid hierarchy')
    return result


def encryption(value, detail):
    detail = detail.lower()
    if detail == 'md5':
        h = hashlib.md5().update(value).digest()
    elif detail == 'sha1':
        h = hashlib.sha1(value).hexdigest()
    elif detail == 'sha224':
        h = hashlib.sha224(value).hexdigest()
    elif detail == 'sha256':
        h = hashlib.sha256(value).hexdigest()
    elif detail == 'sha384':
        h = hashlib.sha384(value).hexdigest()
    elif detail == 'sha512':
        h = hashlib.sha512(value).hexdigest()
    else:
        raise ValueError('Invalid hierarchy')
    return h


def masking(faker_ctx, detail, value):
    label_type = detail['label_type']
    if label_type in faker_ctx:
        faker = faker_ctx[label_type]
        result = faker[value]
    else:
        raise ValueError('Invalid label type {}'.format(label_type))
    return result
