# coding=utf-8



import logging
import sys
import unicodedata
import re

from jinja2 import nodes
from jinja2.ext import Extension
from juicer.exceptions import JuicerException
from six import reraise as raise_
from ast import parse
from gettext import gettext
import json

log = logging.getLogger(__name__)


class HandleExceptionExtension(Extension):
    # a set of names that trigger the extension.
    tags = {'handleinstance'}

    def __init__(self, environment):
        super(HandleExceptionExtension, self).__init__(environment)
        environment.extend()

    def parse(self, parser):
        lineno = next(parser.stream).lineno

        # Retrieves instance
        args = [parser.parse_expression()]
        body = parser.parse_statements(['name:endhandleinstance'],
                                       drop_needle=True)

        result = nodes.CallBlock(self.call_method('_handle', args),
                                 [], [], body).set_lineno(lineno)
        return result

    @staticmethod
    def _handle(instance, caller):
        try:
            return caller()
        except KeyError as ke:
            msg = _('Key error parsing template for instance {instance} {id}. '
                    'Probably there is a problem with port specification') \
                .format(instance=instance.__class__.__name__,
                        id=instance.parameters['task']['id'])
            raise_(lambda: JuicerException(msg + ': ' + str(sys.exc_info()[1])),
                   None, sys.exc_info()[2])
        except TypeError:
            logging.exception(_('Type error in template'))
            msg = _('Type error parsing template for instance {id} '
                    '{instance}.').format(instance=instance.__class__.__name__,
                                          id=instance.parameters['task']['id'])
            raise_(lambda: JuicerException(msg), None, sys.exc_info()[2])


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def convert_variable_name(task_name):
    name = strip_accents(task_name)
    name = re.sub(r"^\s+|\s+$", "", name)
    name = re.sub(r"\s+|\-+", "_", name)
    name = re.sub(r"\_+", "_", name)

    variable_name = ""
    for c in name:
        if c.isalnum() or c is '_':
            variable_name += ''.join(c)

    variable_name = variable_name.lower()

    if is_valid_var_name(variable_name):
        return str(variable_name)
    else:
        raise ValueError(gettext('Parameter task name is invalid.'))


def convert_parents_to_variable_name(parents=[]):
    parents_var_name = []

    for parent in parents:
        name = strip_accents(parent)
        name = re.sub(r"^\s+|\s+$", "", name)
        name = re.sub(r"\s+|\-+", "_", name)
        name = re.sub(r"\_+", "_", name)

        variable_name = ""
        for c in name:
            if c.isalnum() or c is '_':
                variable_name += ''.join(c)

        variable_name = variable_name.lower()

        if is_valid_var_name(variable_name):
            parents_var_name.append(str(variable_name))
        else:
            raise ValueError(gettext('Parameter task name is invalid.'))

    return parents_var_name


def is_valid_var_name(variable_name):
    try:
        parse('{} = None'.format(variable_name))
        return True
    except:
        return False


def kwargs(kwargs_param):
    args = re.sub(r"^\s+|\s+$", "", kwargs_param)
    args = re.sub(r"\s+", " ", args)
    args = re.sub(r"\s*,\s*", ", ", args)
    args = re.sub(r"\s*=\s*", "=", args)

    return args


def get_tuple(field):
    if isinstance(field, tuple):
        return field

    if isinstance(field, str):
        values = field.replace('(', '').replace(')', '').replace(' ', '')\
            .split(',')
        try:
            return tuple([int(v) for v in values])
        except:
            try:
                if len(values) > 1:
                    return tuple([str(v) for v in values])
                return str(field)
            except:
                return None


def get_int_or_tuple(field):
    if field is not None:
        if isinstance(field, str) and field.strip():
            try:
                return int(field)
            except:
                try:
                    values = re.sub(r"\s+|\(|\)|\[|\]|\{|\}", "", field)\
                        .split(',')
                    final_tuple = []
                    for v in values:
                        try:
                            final_tuple.append(int(v))
                        except:
                            final_tuple.append(str(v))
                    return tuple(final_tuple)
                    #return tuple([int(v) for v in values])
                except:
                    return False
    return None


def string_to_list(field):
    tmp_field = re.sub(r"\s+|\[|\]|\(|\)|\{|\}", "", field).split(',')
    field = []

    for item in tmp_field:
        try:
            if '.' in item or 'e' in item:
                field.append(float(item))
            else:
                field.append(int(item))
        except:
            return None

    return field


def string_to_dictionary(field):
    try:
        return json.loads(field)
    except:
        return None


def string_to_int_float_list(field):
    if '.' in field:
        try:
            return float(field)
        except:
            return None
    else:
        try:
            return int(field)
        except:
            try:
                field = re.sub(r"\s+|\[|\]|\(|\)|\{|\}", "", field).split(',')
                field_list = []
                for item in field_list:
                    if '.' in item or 'e' in item:
                        field_list.append(float(item))
                    else:
                        field_list.append(int(item))
            except:
                return None


def rescale(field):
    field = field.replace(',', '.')
    try:
        field = float(field)
    except:
        fields = field.split('/')

        if len(fields) == 2:
            return float(fields[0]) / float(fields[1])

        return None


def tuple_of_tuples(field):
    final_tuple = []

    if field is not None:
        if isinstance(field, str) and field.strip():
            try:
                return int(field)
            except:
                field = re.sub(r"\s+|,$", "", field)
                field = re.sub(r"\{|\[", "(", field)
                field = re.sub(r"\}|\]", ")", field)

                if not field.count("(") == field.count(")"):
                    return False

                values = re.sub(r"\),", "-", field).split('-')
                if len(values) > 1:
                    try:
                        for v in values:
                            in_tuple = re.sub(r"\(|\)", "", v).split(',')
                            tmp_tuple = []
                            for t in in_tuple:
                                try:
                                    tmp_tuple.append(int(t))
                                except:
                                    if t:
                                        tmp_tuple.append(str(t))
                            final_tuple.append(tuple(tmp_tuple))
                        return tuple(final_tuple)
                    except:
                        return False
                else:
                    try:
                        values = re.sub(r"\)|\(", "", values[-1]).split(',')
                        for v in values:
                            try:
                                final_tuple.append(int(v))
                            except:
                                if v:
                                    final_tuple.append(str(v))
                        return tuple(final_tuple)
                    except:
                        return False
    return None


def convert_to_list(field):
    if field is not None:
        if isinstance(field, str) and field.strip():
            field = re.sub(r"\{|\[|\}|\]|\s+", "", field)
            if len(field.split(',')) > 0:
                return True
    return False


def get_random_interval(field):
    if field is not None:
        if isinstance(field, str) and field.strip():
            field = re.sub(r"\{|\[|\}|\]|\(|\)|\s+", "", field)
            field = field.split(',')
            if len(field) == 2:
                return ', '.join(field)
    return False


def get_interval(field):
    if field is not None:
        if isinstance(field, str) and field.strip():
            field = re.sub(r"\{|\[|\}|\]|\(|\)|\s+", "", field)
            field = field.split(':')
            if len(field) == 2:
                return ':'.join(field)
    return False

