# -*- coding: utf-8 -*-
from gettext import gettext
from textwrap import dedent

from juicer.operation import Operation
from juicer.scikit_learn.expression import Expression


class AddColumnsOperation(Operation):
    """
    Merge two data frames, column-wise, similar to the command paste in Linux.
    """
    ALIASES_PARAM = 'aliases'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        self.has_code = len(named_inputs) == 2

        self.suffixes = parameters.get(self.ALIASES_PARAM, '_ds0,_ds1')
        self.suffixes = [s for s in self.suffixes.replace(" ", "").split(',')]

        if not self.has_code:
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")
                .format('input data 1',  'input data  2', self.__class__))

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):

        code = """
        {out} = pd.merge({input1}, {input2}, left_index=True, 
            right_index=True, suffixes=('{s1}', '{s2}')) 
        """.format(out=self.output,
                   s1=self.suffixes[0],
                   s2=self.suffixes[1],
                   input1=self.named_inputs['input data 1'],
                   input2=self.named_inputs['input data 2'])
        return dedent(code)


class AggregationOperation(Operation):
    """
    Computes aggregates and returns the result as a DataFrame.
    Parameters:
        - Expression: a single dict mapping from string to string, then the key
        is the column to perform aggregation on, and the value is the aggregate
        function. The available aggregate functions are avg, max, min, sum,
        count.
    """

    ATTRIBUTES_PARAM = 'attributes'
    FUNCTION_PARAM = 'function'
    PIVOT_ATTRIBUTE = 'pivot'
    PIVOT_VALUE_ATTRIBUTE = 'pivot_values'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        self.attributes = parameters.get(self.ATTRIBUTES_PARAM, [])
        self.functions = parameters.get(self.FUNCTION_PARAM)
        self.input_aliases = {}
        self.input_operations = {}

        self.has_code = len(self.named_inputs) == 1 and any(
                [len(self.named_outputs) == 1, self.contains_results()])

        # Attributes are optional
        self.group_all = len(self.attributes) == 0

        if not all([self.FUNCTION_PARAM in parameters, self.functions]):
            raise ValueError(
                    _("Parameter '{}' must be informed for task {}").format(
                            self.FUNCTION_PARAM, self.__class__))

        for f in parameters[self.FUNCTION_PARAM]:
            if not all([f.get('attribute'), f.get('f'), f.get('alias')]):
                raise ValueError(_('Missing parameter in aggregation function'))

        for dictionary in self.functions:
            att = dictionary['attribute']
            agg = dictionary['f']
            alias = dictionary['alias']
            if att in self.input_operations:
                    self.input_operations[att].append(agg)
                    self.input_aliases[att].append(alias)
            else:
                    self.input_operations[att] = [agg]
                    self.input_aliases[att] = [alias]

        self.values = [agg for agg in self.input_operations]
        self.input_operations = str(self.input_operations)\
            .replace("u'collect_set'", '_merge_set')\
            .replace("u'collect_list'", '_collect_list')

        # noinspection PyArgumentEqualDefault
        self.pivot = parameters.get(self.PIVOT_ATTRIBUTE, None)

        self.pivot_values = parameters.get(self.PIVOT_VALUE_ATTRIBUTE, None)
        self.output = self.named_outputs.get(
                'output data', 'data_{}'.format(self.order))

    def get_data_out_names(self, sep=','):
        return self.output

    def get_output_names(self, sep=", "):
        return self.output

    def generate_code(self):

        code = """
            def _collect_list(x):
                return x.tolist()
            
            def _merge_set(x):
                return set(x.tolist())
            """
        if self.pivot:
            self.pivot = self.pivot[0]

            if self.pivot_values:
                code += """
            values = {values}
            {input} = {input}[{input}['{pivot}'].isin(values)]""".format(
                        output=self.output, values=self.pivot_values,
                        input=self.named_inputs['input data'],
                        pivot=self.pivot)

            code += """
            aggfunc = {aggfunc}
            {output} = pd.pivot_table({input}, index={index}, values={values},
                                      columns=['{pivot}'], aggfunc=aggfunc)
            # rename columns and convert to DataFrame
            {output}.reset_index(inplace=True)
            new_idx = [n[0] if n[1] is ''
                       else "%s_%s_%s" % (n[0],n[1], n[2])
                       for n in {output}.columns.ravel()]    
            {output} = pd.DataFrame({output}.to_records())
            {output}.reset_index(drop=True, inplace=True)
            {output} = {output}.drop(columns='index')
            {output}.columns = new_idx
            """.format(pivot=self.pivot, values=self.values,
                       index=self.attributes, output=self.output,
                       input=self.named_inputs['input data'],
                       aggfunc=self.input_operations)
        else:
            code += """
            columns = {columns}
            target = {aliases}
            operations = {operations}
    
            {output} = {input}.groupby(columns).agg(operations)
            new_idx = []
            i = 0
            old = None
            for (n1, n2) in {output}.columns.ravel():
                if old != n1:
                    old = n1
                    i = 0
                new_idx.append(target[n1][i])
                i += 1
        
            {output}.columns = new_idx
            {output} = {output}.reset_index()
            {output}.reset_index(drop=True, inplace=True)
            """.format(output=self.output,
                       input=self.named_inputs['input data'],
                       columns=self.attributes,
                       aliases=self.input_aliases,
                       operations=self.input_operations)
        return dedent(code)


class CleanMissingOperation(Operation):
    """
    Clean missing fields from data set
    Parameters:
        - attributes: list of attributes to evaluate
        - cleaning_mode: what to do with missing values. Possible values include
          * "VALUE": replace by parameter "value",
          * "MEDIAN": replace by median value
          * "MODE": replace by mode value
          * "MEAN": replace by mean value
          * "REMOVE_ROW": remove entire row
          * "REMOVE_COLUMN": remove entire column
        - value: optional, used to replace missing values
    """
    ATTRIBUTES_PARAM = 'attributes'
    MIN_MISSING_RATIO_PARAM = 'min_missing_ratio'
    MAX_MISSING_RATIO_PARAM = 'max_missing_ratio'
    CLEANING_MODE_PARAM = 'cleaning_mode'
    VALUE_PARAMETER = 'value'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        self.has_code = len(self.named_inputs) == 1

        if self.has_code:
            if self.ATTRIBUTES_PARAM in parameters:
                self.attributes_CM = parameters[self.ATTRIBUTES_PARAM]
            else:
                raise ValueError(
                    _("Parameter '{}' must be informed for task {}")
                    .format('attributes', self.__class__))

            self.min_ratio = abs(parameters.get(
                    self.MIN_MISSING_RATIO_PARAM, 0.0))
            self.max_ratio = abs(parameters.get(
                    self.MAX_MISSING_RATIO_PARAM, 1.0))

            if any([self.min_ratio > self.max_ratio,
                    self.min_ratio > 1.0,
                    self.max_ratio > 1.0]):
                raise ValueError(
                        _("Parameter '{}' must be 0<=x<=1 for task {}").format(
                                'attributes', self.__class__))

            self.mode_CM = self.parameters.get(self.CLEANING_MODE_PARAM,
                                               "REMOVE_ROW")
            self.value_CM = self.parameters.get(self.VALUE_PARAMETER, None)

            if all([self.value_CM is None,  self.mode_CM == "VALUE"]):
                raise ValueError(
                        _("Parameter '{}' must be not None when"
                          " mode is '{}' for task {}").format(
                                self.VALUE_PARAMETER, 'VALUE', self.__class__))

            self.output = self.named_outputs.get(
                'output result', 'output_data_{}'.format(self.order))

    def generate_code(self):

        op = ""

        if self.mode_CM == "REMOVE_ROW":
            op = "{output}.dropna(subset=col, axis='index', inplace=True)"\
                .format(output=self.output)
        elif self.mode_CM == "REMOVE_COLUMN":
            op = "{output}[col].dropna(axis='columns', inplace=True))"\
                .format(output=self.output)

        elif self.mode_CM == "VALUE":
            op = "{output}[col].fillna(value={value}, inplace=True)"\
                .format(output=self.output, value=self.value_CM)

        elif self.mode_CM == "MEAN":
            op = "{output}[col].fillna(value={output}" \
                 "[col].mean(), inplace=True)".format(output=self.output)
        elif self.mode_CM == "MEDIAN":
            op = "{output}[col].fillna(value={output}" \
                 "[col].median(), inplace=True)".format(output=self.output)

        elif self.mode_CM == "MODE":
            op = "{out}[col].fillna(value={out}[col].mode(), inplace=True)"\
                .format(out=self.output)

        code = """
        min_missing_ratio = {min_thresh}
        max_missing_ratio = {max_thresh}
        {output} = {input}
        for col in {columns}:
            ratio = {input}[col].isnull().sum()
            if ratio >= min_missing_ratio and ratio <= max_missing_ratio:
                {op}
            """\
            .format(min_thresh=self.min_ratio, max_thresh=self.max_ratio,
                    output=self.output, input=self.named_inputs['input data'],
                    columns=self.attributes_CM, op=op)
        return dedent(code)


class DifferenceOperation(Operation):
    """
    Returns a new DataFrame containing rows in this frame but not in another
    frame.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(named_inputs) == 2
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = """
        cols = {input1}.columns
        {output} = pd.merge({input1}, {input2},
            indicator=True, how='left', on=None)
        {output} = {output}.loc[{output}['_merge'] == 'left_only', cols]
        """.format(output=self.output,
                   input1=self.named_inputs['input data 1'],
                   input2=self.named_inputs['input data 2'])
        return dedent(code)


class DistinctOperation(Operation):
    """
    Returns a new DataFrame containing the distinct rows in this DataFrame.
    Parameters: attributes to consider during operation (keys)
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.attributes = parameters.get('attributes', 'None')
        self.has_code = len(named_inputs) == 1
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = "{out} = {in1}.drop_duplicates(subset={columns}, keep='first')"\
            .format(out=self.output, in1=self.named_inputs['input data'],
                    columns=self.attributes)
        return dedent(code)


class DropOperation(Operation):
    """
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    The only parameters is the name of the columns to be removed.
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
            self.cols = ','.join(['"{}"'.format(x)
                                   for x in self.attributes])
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.has_code = len(named_inputs) == 1
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = "{output} = {input}.drop(columns={columns})"\
            .format(output=self.output, input=self.named_inputs['input data'],
                    columns=self.attributes)
        return dedent(code)


class ExecutePythonOperation(Operation):
    PYTHON_CODE_PARAM = 'code'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if not all([self.PYTHON_CODE_PARAM in parameters]):
            msg = _("Required parameter {} must be informed for task {}")
            raise ValueError(msg.format(self.PYTHON_CODE_PARAM, self.__class__))

        self.code = parameters.get(self.PYTHON_CODE_PARAM)

        # Always execute
        self.has_code = True
        self.out1 = self.named_outputs.get('output data 1',
                                           'out_1_{}'.format(self.order))
        self.out2 = self.named_outputs.get('output data 2',
                                           'out_2_{}'.format(self.order))

    def get_output_names(self, sep=", "):
        return sep.join([self.out1, self.out2])

    def generate_code(self):
        in1 = self.named_inputs.get('input data 1', 'None')

        in2 = self.named_inputs.get('input data 2', 'None')

        code = dedent("""
        import json
        from RestrictedPython.Guards import safe_builtins
        from RestrictedPython.RCompile import compile_restricted
        from RestrictedPython.PrintCollector import PrintCollector

        results = [r[1].result() for r in task_futures.items() if r[1].done()]
        results = dict([(r['task_name'], r) for r in results])
        # Input data
        in1 = {in1}
        in2 = {in2}

        # Output data, initialized as None
        out1 = None
        out2 = None

        # Variables and language supported
        ctx = {{
            'wf_results': results,
            'in1': in1,
            'in2': in2,
            'out1': out1,
            'out2': out2,
            
            # Restrictions in Python language
             '_write_': lambda v: v,
            '_getattr_': getattr,
            '_getitem_': lambda ob, index: ob[index],
            '_getiter_': lambda it: it,
            '_print_': PrintCollector,
            'json': json,
        }}
        user_code = \"\"\"{code}\"\"\"

        ctx['__builtins__'] = safe_builtins

        compiled_code = compile_restricted(user_code,
        str('python_execute_{order}'), str('exec'))
        try:
            exec compiled_code in ctx

            # Retrieve values changed in the context
            out1 = ctx['out1']
            out2 = ctx['out2']

            if '_print' in ctx:
                emit_event(name='update task',
                    message=ctx['_print'](),
                    status='RUNNING',
                    identifier='{id}')
        except NameError as ne:
            raise ValueError(_('Invalid name: {{}}. '
                'Many Python commands are not available in Lemonade').format(ne))
        except ImportError as ie:
            raise ValueError(_('Command import is not supported'))
        """.format(in1=in1, in2=in2, code=self.code.encode('unicode_escape'),
                   name="execute_python", order=self.order,
                   id=self.parameters['task']['id']))

        code += dedent("""
        {out1} = out1
        {out2} = out2
        """.format(out1=self.out1, out2=self.out2))
        return dedent(code)


class ExecuteSQLOperation(Operation):
    QUERY_PARAM = 'query'
    NAMES_PARAM = 'names'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)

        if not all([self.QUERY_PARAM in parameters]):
            msg = _("Required parameter {} must be informed for task {}")
            raise ValueError(msg.format(self.QUERY_PARAM, self.__class__))

        self.query = ExecuteSQLOperation._escape_string(
            parameters.get(self.QUERY_PARAM).strip().replace('\n', ' '))
        if self.query[:6].upper() != 'SELECT':
            raise ValueError(_('Invalid query. Only SELECT is allowed.'))

        if self.NAMES_PARAM in parameters:
            self.names = [
                n.strip() for n in parameters.get(self.NAMES_PARAM).split(',')
                if n.strip()]
        else:
            self.names = None

        self.has_code = any([len(self.named_outputs) > 0,
                             self.contains_results()])
        self.input1 = self.named_inputs.get('input data 1')
        self.input2 = self.named_inputs.get('input data 2')
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))

        self.has_import = 'from pandasql import sqldf\n'

    def get_data_out_names(self, sep=','):
        return self.output

    @staticmethod
    def _escape_string(value):
        """ Escape a SQL string. Borrowed from
        https://github.com/PyMySQL/PyMySQL/blob/master/pymysql/converters.py"""
        return value
        # _escape_table = [unichr(x) for x in range(128)]
        # _escape_table[0] = u'\\0'
        # _escape_table[ord('\\')] = u'\\\\'
        # _escape_table[ord('\n')] = u'\\n'
        # _escape_table[ord('\r')] = u'\\r'
        # _escape_table[ord('\032')] = u'\\Z'
        # _escape_table[ord('"')] = u'\\"'
        # _escape_table[ord("'")] = u"\\'"
        # return value.translate(_escape_table)

    def generate_code(self):
        code = dedent(u"""

        query = {query}
        {out} = sqldf(query, {{'ds1': {in1}, 'ds2': {in2}}})
        names = {names}
        
        if names is not None and len(names) > 0:
            old_names = {out}.columns
            if len(old_names) != len(names):
                raise ValueError('{invalid_names}')
            rename = dict(zip(old_names, names))
            {out}.rename(columns=rename, inplace=True)
        """.format(in1=self.input1, in2=self.input2, query=repr(self.query),
                   out=self.output, names=repr(self.names),
                   invalid_names=_('Invalid names. Number of attributes in '
                                   'result differs from names informed.')))
        return code


class FilterOperation(Operation):
    """
    Filters rows using the given condition.
    Parameters:
        - The expression (==, <, >)
    """
    FILTER_PARAM = 'filter'
    ADVANCED_FILTER_PARAM = 'expression'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if self.FILTER_PARAM not in parameters and self.ADVANCED_FILTER_PARAM \
                not in parameters:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}".format(
                    self.FILTER_PARAM, self.__class__)))

        self.advanced_filter = parameters.get(self.ADVANCED_FILTER_PARAM) or []
        self.filter = parameters.get(self.FILTER_PARAM) or []

        self.has_code = any(
            [len(self.named_inputs) == 1, self.contains_results()])
        self.output = self.named_outputs.get('output data',
                                             'out_{}'.format(self.order))

    def generate_code(self):
        input_data = self.named_inputs['input data']
        params = {'input': input_data}

        filters = [
            "({0} {1} {2})".format(f['attribute'], f['f'],
                                   f.get('value', f.get('alias')))
            for f in self.filter]

        code = """
        {out} = {input}""".format(out=self.output,
                                  input=self.named_inputs['input data'])

        expressions = []
        for i, expr in enumerate(self.advanced_filter):
            expression = Expression(expr['tree'], params)
            expressions.append(expression.parsed_expression)

        if len(expressions) > 0:
            for e in expressions:
                code += """
        {out} = {out}[{out}.apply({expr}, axis=1)]""".format(out=self.output,
                                                             expr=e)

        indentation = " and "
        if len(filters) > 0:
            code += """
        {out} = {out}.query('{f}')""".format(out=self.output,
                                             f=indentation.join(filters))

        return dedent(code)


class IntersectionOperation(Operation):
    """
    Returns a new DataFrame containing rows only in both this
    frame and another frame.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(named_inputs) == 2

        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                .format('input data 1',  'input data 2', self.__class__))

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = """
        if len({in1}.columns) != len({in2}.columns):
            raise ValueError('{error}')
        {in1} = {in1}.dropna(axis=0, how='any')
        {in2} = {in2}.dropna(axis=0, how='any')
        keys = {in1}.columns.tolist()
        {in1} = pd.merge({in1}, {in2}, how='left', on=keys, 
        indicator=True, copy=False)
        {out} = {in1}.loc[{in1}['_merge'] == 'both', keys]
        """.format(out=self.output,
                   in1=self.named_inputs['input data 1'],
                   in2=self.named_inputs['input data 2'],
                   error='For intersection operation, both input data '
                         'sources must have the same number of attributes '
                         'and types.')
        return dedent(code)


class JoinOperation(Operation):
    """
    Joins with another DataFrame, using the given join expression.
    The expression must be defined as a string parameter.
    """
    KEEP_RIGHT_KEYS_PARAM = 'keep_right_keys'
    MATCH_CASE_PARAM = 'match_case'
    JOIN_TYPE_PARAM = 'join_type'
    LEFT_ATTRIBUTES_PARAM = 'left_attributes'
    RIGHT_ATTRIBUTES_PARAM = 'right_attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.keep_right_keys = \
            parameters.get(self.KEEP_RIGHT_KEYS_PARAM, False) in (1, '1', True)
        self.match_case = parameters.get(self.MATCH_CASE_PARAM, False) \
                          in (1, '1', True)
        self.join_type = parameters.get(self.JOIN_TYPE_PARAM, 'inner')

        self.join_type = self.join_type.replace("_outer", "")

        if not all([self.LEFT_ATTRIBUTES_PARAM in parameters,
                    self.RIGHT_ATTRIBUTES_PARAM in parameters]):
            raise ValueError(
                _("Parameters '{}' and '{}' must be informed for task {}")\
                    .format(self.LEFT_ATTRIBUTES_PARAM,
                            self.RIGHT_ATTRIBUTES_PARAM,
                            self.__class__))

        self.has_code = len(named_inputs) == 2
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                    .format('input data 1',  'input data 2', self.__class__))

        self.left_attributes = parameters.get(self.LEFT_ATTRIBUTES_PARAM)
        self.right_attributes = parameters.get(self.RIGHT_ATTRIBUTES_PARAM)

        self.suffixes = parameters.get('aliases', '_l,_r')
        self.suffixes = [s for s in self.suffixes.split(',')]
        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(
                                                     self.order))

    def generate_code(self):

        code = ""
        if not self.keep_right_keys:
            code = """
            cols_to_remove = [c+'{sf}' for c in 
                {in2}.columns if c in {in1}.columns]
            """.format(in1=self.named_inputs['input data 1'],
                       in2=self.named_inputs['input data 2'],
                       sf=self.suffixes[1])

        if self.match_case:
            code += """
            data1_tmp = {in1}[{id1}].applymap(lambda col: str(col).lower())
            data1_tmp.columns = [c+"_lower" for c in data1_tmp.columns]
            data1_tmp = pd.concat([{in1}, data1_tmp], axis=1, sort=False)
            
            data2_tmp = {in2}[{id2}].applymap(lambda col: str(col).lower())
            data2_tmp.columns = [c+"_lower" for c in data2_tmp.columns]
            data2_tmp = pd.concat([{in2}, data2_tmp], axis=1, sort=False)

            {out} = pd.merge(data1_tmp, data2_tmp, left_on=col1, right_on=col2,
                copy=False, suffixes={suffixes}, how='{type}')
            {out}.drop(col1+col2, axis=1, inplace=True)
             """.format(out=self.output, type=self.join_type,
                        in1=self.named_inputs['input data 1'],
                        in2=self.named_inputs['input data 2'],
                        id1=self.left_attributes,
                        id2=self.right_attributes,
                        suffixes=self.suffixes)
        else:
            code += """
            {out} = pd.merge({in1}, {in2}, how='{type}', 
                suffixes={suffixes},
                left_on={id1}, right_on={id2})
             """.format(out=self.output, type=self.join_type,
                        in1=self.named_inputs['input data 1'],
                        in2=self.named_inputs['input data 2'],
                        id1=self.left_attributes,
                        id2=self.right_attributes,
                        suffixes=self.suffixes)

        if not self.keep_right_keys:
            code += """
            {out}.drop(cols_to_remove, axis=1, inplace=True)
            """.format(out=self.output)

        return dedent(code)


class ReplaceValuesOperation(Operation):
    """
    Replace values in one or more attributes from a dataframe.
    Parameters:
    - The list of columns selected.
    """
    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        self.replaces = {}

        if any(['value' not in parameters,
                'replacement' not in parameters]):
            raise ValueError(
                _("Parameter {} and {} must be informed if is using "
                  "replace by value in task {}.").format(
                        'value', 'replacement', self.__class__))

        for att in parameters['attributes']:
            if att not in self.replaces:
                self.replaces[att] = [[], []]
            self.replaces[att][0].append(self.parameters['value'])
            self.replaces[att][1].append(self.parameters['replacement'])

        self.has_code = len(named_inputs) == 1
        self.output = self.named_outputs.get('output data',
                                             'output_data_{}'.format(self.order))

    def generate_code(self):
        code = """
        {out} = {in1}
        replacement = {replaces}
        for col in replacement:
            list_replaces = replacement[col]
            {out}[col] = {out}[col].replace(list_replaces[0], list_replaces[1])
        """.format(out=self.output, in1=self.named_inputs['input data'],
                   replaces=self.replaces)
        return dedent(code)


class SampleOrPartitionOperation(Operation):
    """
    Returns a sampled subset of this DataFrame.
    Parameters:
    - fraction -> fraction of the data frame to be sampled.
        without replacement: probability that each element is chosen;
            fraction must be [0, 1]
    - seed -> seed for random operation.
    """
    TYPE_VALUE = 'value'
    TYPE_PERCENT = 'percent'
    TYPE_HEAD = 'head'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.type = self.parameters.get('type', self.TYPE_PERCENT)
        self.value = int(self.parameters.get('value', -1))
        self.fraction = float(self.parameters.get('fraction', -50)) / 100

        if self.value < 0 and self.type != self.TYPE_PERCENT:
            raise ValueError(
                _("Parameter 'value' must be [x>=0] if is using "
                  "the current type of sampling in task {}.")
                .format(self.__class__))
        if self.type == self.TYPE_PERCENT and any([self.fraction > 1.0,
                                                   self.fraction < 0]):
            raise ValueError(
                    _("Parameter 'fraction' must be 0<=x<=1 if is using "
                      "the current type of sampling in task {}.")
                        .format(self.__class__))

        self.seed = self.parameters.get('seed', 'None')
        self.seed = self.seed if self.seed != "" else 'None'

        self.output = self.named_outputs.get('sampled data',
                                             'output_data_{}'.format(
                                                     self.order))

        self.has_code = len(self.named_inputs) == 1

    def generate_code(self):
        if self.type == self.TYPE_PERCENT:
            code = """
            {output} = {input}.sample(frac={value}, random_state={seed})
            """.format(output=self.output,
                       input=self.named_inputs['input data'],
                       seed=self.seed, value=self.fraction)
        elif self.type == self.TYPE_HEAD:
            code = """
            {output} = {input}.head({value})
            """.format(output=self.output,
                       input=self.named_inputs['input data'], value=self.value)
        else:
            code = """
            {output} = {input}.sample(n={value}, random_state={seed})
            """.format(output=self.output,
                       input=self.named_inputs['input data'],
                       seed=self.seed, value=self.value)

        return dedent(code)


class SelectOperation(Operation):
    """
    Projects a set of expressions and returns a new DataFrame.
    Parameters:
    - The list of columns selected.

    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        if self.ATTRIBUTES_PARAM in parameters:
            self.attributes = parameters.get(self.ATTRIBUTES_PARAM)
            self.cols = ','.join(['"{}"'.format(x)
                                  for x in self.attributes])
        else:
            raise ValueError(
                _("Parameter '{}' must be informed for task {}").format(
                    self.ATTRIBUTES_PARAM, self.__class__))

        self.has_code = len(named_inputs) == 1
        self.output = self.named_outputs.get(
            'output projected data', 'projection_data_{}'.format(self.order))

    def generate_code(self):

        code = "{output} = {input}[[{column}]]"\
            .format(output=self.output, column=self.cols,
                    input=self.named_inputs['input data'])
        return dedent(code)


class SortOperation(Operation):
    """
    Returns a new DataFrame sorted by the specified column(s).
    Parameters:
    - The list of columns to be sorted.
    - A list indicating whether the sort order is ascending for the columns.
    Condition: the list of columns should have the same size of the list of
               boolean to indicating if it is ascending sorting.
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs, named_outputs)

        attributes = parameters.get(self.ATTRIBUTES_PARAM, '')
        if len(attributes) == 0:
            raise ValueError(
                    gettext("Parameter '{}' must be"
                            " informed for task {}").format(
                            self.ATTRIBUTES_PARAM, self.__class__))

        self.columns = [att['attribute'] for att in attributes]
        self.ascending = [True for _ in range(len(self.columns))]
        for i, v in enumerate([att['f'] for att in attributes]):
            if v != "asc":
                self.ascending[i] = False

        self.has_code = len(named_inputs) == 1
        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = "{out} = {input}.sort_values(by={columns}, ascending={asc})"\
            .format(out=self.output, input=self.named_inputs['input data'],
                    columns=self.columns, asc=self.ascending)
        return dedent(code)


class SplitOperation(Operation):
    """
    Randomly splits a Data Frame into two data frames.
    Parameters:
    - List with two weights for the two new data frames.
    - Optional seed in case of deterministic random operation
        ('0' means no seed).

    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)

        self.has_code = len(named_inputs) == 1

        self.weights = float(self.parameters.get('weights', 50))/100
        self.seed = self.parameters.get("seed", 'None')
        self.seed = 'None' if self.seed == "" else self.seed

        self.out1 = self.named_outputs.get('splitted data 1',
                                           'splitted_1_{}'.format(self.order))
        self.out2 = self.named_outputs.get('splitted data 2',
                                           'splitted_2_{}'.format(self.order))

    def get_data_out_names(self, sep=','):
            return ''

    def get_output_names(self, sep=', '):
        return sep.join([self.out1, self.out2])

    def generate_code(self):
        code = """{out1}, {out2} = np.split({input}.sample(frac=1, 
        random_state={seed}), [int({weights}*len({input}))])
        """.format(out1=self.out1, out2=self.out2,
                   input=self.named_inputs['input data'],
                   seed=self.seed, weights=self.weights)
        return dedent(code)


class TransformationOperation(Operation):
    """
    Returns a new DataFrame applying the expression to the specified column.
    Parameters:
        - Alias: new column name. If the name is the same of an existing,
            replace it.
        - Expression: json describing the transformation expression
    """
    ALIAS_PARAM = 'alias'
    EXPRESSION_PARAM = 'expression'

    def __init__(self, parameters, named_inputs, named_outputs):
        Operation.__init__(self, parameters, named_inputs, named_outputs)
        self.has_code = any(
                [len(self.named_inputs) > 0, self.contains_results()])
        if self.has_code:
            if self.EXPRESSION_PARAM in parameters:
                self.expressions = parameters[self.EXPRESSION_PARAM]
            else:
                msg = _("Parameter must be informed for task {}.")
                raise ValueError(
                        msg.format(self.EXPRESSION_PARAM, self.__class__))
            self.output = self.named_outputs.get(
                    'output data', 'sampled_data_{}'.format(self.order))

    def generate_code(self):
        # Builds the expression and identify the target column
        params = {'input': self.named_inputs['input data']}
        functions = ""
        for expr in self.expressions:
            expression = expr['tree']
            expression = Expression(expression, params)
            f = expression.parsed_expression
            functions += "['{}', {}],".format(expr['alias'], f)
            # row.append(expression.imports) #TODO: by operation itself

        code = """
        {out} = {input}.copy()

        functions = [{expr}]
        for col, function in functions:
            {out}[col] = {out}.apply(function, axis=1)
        """.format(out=self.output,
                   input=self.named_inputs['input data'],
                   expr=functions)
        return dedent(code)


class UnionOperation(Operation):
    """
    Return a new DataFrame containing all rows in this frame and another frame.
    Takes no parameters.
    """

    def __init__(self, parameters,  named_inputs, named_outputs):
        Operation.__init__(self, parameters,  named_inputs,  named_outputs)
        self.has_code = len(self.named_inputs) == 2
        if not self.has_code:
            raise ValueError(
                _("Parameter '{}' and '{}' must be informed for task {}")
                .format('input data 1',  'input data 2', self.__class__))

        self.output = self.named_outputs.get(
            'output data', 'output_data_{}'.format(self.order))

    def generate_code(self):
        code = """
        {0} = pd.concat([{1}, {2}], sort=False, axis=0, ignore_index=True)
        """.format( self.output,
                    self.named_inputs['input data 1'],
                    self.named_inputs['input data 2'])
        return dedent(code)

