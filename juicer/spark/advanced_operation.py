# -*- coding: utf-8 -*-

import logging
import json
import datetime 
from textwrap import dedent
from jinja2 import Environment, BaseLoader
from juicer.operation import Operation
from juicer.spark.data_operation import DataReaderOperation


log = logging.getLogger()
log.setLevel(logging.DEBUG)
class UserFilterOperation(Operation):
    """
    Handles filters that requires user input.
    """
    USE_ADVANCED_EDITOR_PARAM = 'use_advanced_editor' # No use in Juicer
    IGNORE_PARAM = 'ignore' 
    FILTERS_PARAM = 'filters'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs,
                                     named_outputs)

        self.ignore = parameters.get('ignore') in ('1', 1, 'true', True)
        if not self.ignore:
            if self.FILTERS_PARAM in parameters:
                self.filters = UserFilterOperation._parse_filters(
                        parameters.get(self.FILTERS_PARAM))
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}".format(
                        self.FILTERS_PARAM, self.__class__)))

        self.has_code = any(
            [len(self.named_outputs) > 0, self.contains_results()])

    def generate_code(self):
        code_template = dedent("""
        {%- if ignore %}
        # operation is being ignored (property ignore = True)
        {{out}} = {{_input}}
        {%- else %}
        to_date = lambda d: functions.unix_timestamp(
            functions.lit(d)).cast("timestamp")

        {{out}} = {{_input}}.filter(
            {%- for f in filter_conditions %}
            ({{f}}){% if not loop.last%} &{% endif %}
            {%- endfor %})
        {%- endif %}
        """)
        ctx = dict(
            out=self.named_outputs.get(
                'output data', 'out_{}'.format(self.order)),
            _input=self.named_inputs['input data'],
            ignore=self.ignore,
            filter_conditions=self.filters
        )
        template = Environment(loader=BaseLoader).from_string(code_template)
        return template.render(ctx)

    @staticmethod
    def _parse_filters(filters):
        parsed = []
        for f in filters:
            #import pdb; pdb.set_trace()
            if not UserFilterOperation._is_required_and_filled(f):
                raise ValueError(
                    _('Filter "{}" is required and its value was not informed.'
                    ).format(f.get('label')))
            if (f.get('name', '') or '') == '':
                raise ValueError(
                    _('Inconsistency in workflow: filter has no name'))

            # Simplify by using only value
            if (f.get('value', '') or '') == '':
                f['value'] = f.get('default_value') 
            valid, converted = UserFilterOperation._is_value_valid(f)
            if not valid:
                raise ValueError(_(
                    'Invalid value for "{}": "{}". Data type: {} (op: {}).').format(
                        f.get('label'), f.get('value'), f.get('type'), 
                        f.get('operator')))

            f['converted'] = converted
            op = UserFilterOperation._translate_filter(f)
            parsed.append(op)
        return parsed

    @staticmethod
    def _is_required_and_filled(f):
        return not (f.get('multiplicity') in ('ONE', 'ONE_OR_MORE') 
                and (f.get('value', '') or '') == ''
                and (f.get('default_value', '') or '') == ''
               ) 

    @staticmethod
    def _is_value_list_valid(f):
        v = f.get('value')

        if f.get('type') == 'CHARACTER' or f.get('type') is None:
            return True, ', '.join(map(lambda x: '"' + x.strip() + '"', v.split(',')))
        elif f.get('type') == 'INTEGER':
            try:
                # range 1:-1 to remove braces []
                if isinstance(v, (str,)):
                    return True, ', '.join(map(lambda x: str(int(x)), v[1:-1].split(',')))
                elif isinstance(v, list):
                    return True, v
            except:
                return False, None
        elif f.get('type') == 'DECIMAL':
            try:
                return True, ', '.join(map(lambda x: str(float(x)), v.split(',')))
            except ValueError:
                return False, None
        #elif f.get('type') == 'DATE':
        #    # FIXME handle other formats
        #    try:
        #        return True, '[' + ', '.join(map(lambda x: str(float(x)))) + ']'
        #        return (True, 
        #            'to_date("{}")'.format(datetime.datetime.strptime(
        #                        v, '%d/%m/%Y').strftime('%Y-%m-%d 00:00:00')))
            except:
                return False, None

    @staticmethod
    def _is_value_valid(f):
        v = f.get('value')
        if f.get('operator') in ('in', 'ni'):
            return UserFilterOperation._is_value_list_valid(f)
        v = f.get('value')
        if f.get('type') == 'CHARACTER' or f.get('type') is None:
            return True, '"' + v + '"'
        elif f.get('type') == 'INTEGER':
            try:
                return True, int(v)
            except:
                return False, None
        elif f.get('type') == 'DECIMAL':
            try:
                return True, float(v)
            except ValueError:
                return False, None
        elif f.get('type') == 'DATE':
            # FIXME handle other formats
            try:
                return (True, 
                    'to_date("{}")'.format(datetime.datetime.strptime(
                                v, '%Y-%m-%d').strftime('%Y-%m-%d 00:00:00')))
            except:
                try:
                    return (True, 
                        'to_date("{}")'.format(datetime.datetime.strptime(
                            v, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')))
                except:
                    return False, None
        log.warn(_('Unknown data type %s for field %s', f.get('name'), 
            f.get('type')))
        return False, None

    @staticmethod
    def _translate_filter(f):
        value = f.get('converted')
        name = f.get('name')
        op = f.get('operator', '')

        if op == 'eq':
            return f'functions.col("{name}") == {value}'
        elif op == 'ne':
            return f'functions.col("{name}") != {value}'
        elif op == 'gt':
            return f'functions.col("{name}") > {value}'
        elif op == 'lt':
            return f'functions.col("{name}") < {value}'
        elif op == 'gt':
            return f'functions.col("{name}") > {value}'
        elif op == 'ge':
            return f'functions.col("{name}") >= {value}'
        elif op == 'le':
            return f'functions.col("{name}") <= {value}'
        elif op == 'in':
            return f'functions.col("{name}").isin(*[{value}])'
        elif op == 'ni':
            return f'~functions.col("{name}").isin(*[{value}])'
 


