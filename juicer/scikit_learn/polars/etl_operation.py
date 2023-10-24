# -*- coding: utf-8 -*-
from gettext import gettext
from textwrap import dedent

from juicer.scikit_learn.polars.expression import (
    JAVA_2_PYTHON_DATE_FORMAT, Expression)
import juicer.scikit_learn.etl_operation as sk


class AddColumnsOperation(sk.AddColumnsOperation):
    """
    Merge two data frames, column-wise, similar to the command paste in Linux.
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return ''
        input1 = self.named_inputs['input data 1']
        input2 = self.named_inputs['input data 2']
        s1 = self.suffixes[0]
        s2 = self.suffixes[1]

        # Ok
        code = f"""
            # Get all unique column names in both dataframes
            ok = set({input1}.columns).symmetric_difference(
                set({input2}.columns))

            alias1 = []
            for c in {input1}.columns:
                if c in ok:
                    alias1.append(pl.col(c))
                else:
                    alias1.append(pl.col(c).alias(f'{{c}}{s1}'))
            alias2 = []
            for c in {input2}.columns:
                if c in ok:
                    alias2.append(pl.col(c))
                else:
                    alias2.append(pl.col(c).alias(f'{{c}}{s2}'))
            # Lazy only allows {'vertical', 'diagonal'} concat strategy in 0.15.6
            {self.output} = pl.concat([
                {input1}.select(alias1).collect(),
                {input2}.select(alias2).collect()],
                how='horizontal').lazy()
        """
        return dedent(code)


class AggregationOperation(sk.AggregationOperation):
    """
    Computes aggregates and returns the result as a DataFrame.
    Parameters:
        - Expression: a single dict mapping from string to string, then the key
        is the column to perform aggregation on, and the value is the aggregate
        function. The available aggregate functions avg, collect_list,
        collect_set, count, first, last, max, min, sum and size.
    Since 2.6
    """

    template = """
    {%- if op.pivot[0] %}
        {%- if op.pivot_values %}
    {{op.output}} = {{op.input}}.filter(
            pl.col('{{op.pivot[0]}}').isin([{{op.values}}])
        {%- else %}
    {{op.output}} = {{op.input}}
        {%- endif %}
    {{op.output}} = ({{op.output}}
        .groupby({{op.attributes}} + {{op.pivot}})
        .agg([
            {%- for f in op.functions %}
            pl.col('{{f.attribute}}').{{f.f}}
                {%- if not f.f.endswith(')') %}(){% endif -%}
                .alias('{{f.alias}}'),
            {%- endfor %}
        ])
        .collect()
        .pivot(index={{op.attributes}}, values=[
            {%- for f in op.functions %}'{{f.alias}}',{% endfor %}],
            aggregate_fn='min', columns={{op.pivot}}).lazy()
    )
    {%- else %}
    {{op.output}} = ({{op.input}}.groupby(
        {{op.attributes}}).agg([
            {%- for f in op.functions %}
            pl.col('{{f.attribute}}').{{f.f}}
                {%- if not f.f.endswith(')') %}(){% endif -%}
                .alias('{{f.alias}}'),
            {%- endfor %}
    ]))
    {%- endif %}
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

        mapping_names = {
            'avg': 'mean',
            'skewness': 'skew',
            'collect_list': 'list',
            'collect_set': 'apply(lambda x: set(x.drop_nulls()))',
            'countDistinct': 'unique().count()',
            'approx_count_distinct': 'unique().count()',
            'sumDistinct': 'unique().sum()',
            'stddev': 'std',
            'stddev_pop':
                'apply(lambda x: np.std(x.drop_nulls().to_numpy(), ddof=0))',
            'variance': 'var',
            'var_pop':
                'apply(lambda x: np.var(x.drop_nulls().to_numpy(), ddof=0))',
        }
        for f in self.functions:
            f['f'] = mapping_names.get(f['f'], f['f'])

    def generate_code(self):
        if not self.has_code:
            return None
        self.input = self.named_inputs['input data']
        return dedent(self.render_template({'op': self}))


class CastOperation(sk.CastOperation):
    """ Change attribute type.
    Since 2.6
    """

    template = """
        try:
            strict = {{op.errors == 'raise'}}
            {{op.output}} = {{op.input}}.with_columns([
        {%- for attr in op.attributes %}
            {%- if attr.type in ('Boolean', 'Date', 'Time', 'Datetime',
                                'UInt8', 'UInt16', 'UInt32', 'UInt64',
                                'Int8', 'Int16', 'Int32', 'Int64',
                                'Float32', 'Float64', 'Categorical') %}
                {%- set method_call = 'cast(pl.' + attr.type +', strict=strict)' %}
            {%- elif attr.type == 'Text' %}
                {%- set method_call = 'cast(pl.Utf8)' %}
            {%- elif attr.type == 'List' %}
                {%- set method_call = 'apply(lambda x: [x])' %}
            {%-endif %}
            {%- if op.errors == 'move' %}
            pl.when(pl.col('{{attr.attribute}}').{{method_call}}.is_null() &
                pl.col('{{attr.attribute}}').is_not_null()).then(
                    pl.col('{{attr.attribute}}')
                ).alias('error_cast_{{attr.attribute}}'),
            {%- endif %}
            pl.col('{{attr.attribute}}').{{method_call}},
        {%- endfor %}
            ]).lazy()

        except ValueError as ve:
            msg = str(ve)
            if 'Unable to parse string' in msg:
                expr = re.compile(r'.+string "(.+)" at position (\\d+)')
                parts = expr.findall(msg)[0]
                raise ValueError('{{errors.unable_to_parse}}'.format(*parts))
            else:
                raise
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        errors = {
            'NaN_2_int': gettext(
                'Integer values cannot be null. '
                'Handle them or use a Decimal type.'),
            'unable_to_parse': gettext(
                'Unable to convert value {} at record {} (starts from 0).')
        }
        if self.has_code:
            return dedent(self.render_template({'op': self, 'errors': errors}))


class CleanMissingOperation(sk.CleanMissingOperation):
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
    Since 2.6
    """
    template = """
    {%- if mode == "REMOVE_ROW" %}
        cols = {{columns}}
        min_missing_ratio = {{min_thresh}}
        max_missing_ratio = {{max_thresh}}
        {{output}} = ({{input}}.with_column(
                pl.sum(pl.col(cols).is_null()).alias('@res'))
            .filter(pl.col('@res').is_between(
                min_missing_ratio, max_missing_ratio, (False, True)))
            .select(pl.exclude('@res'))
        )

    {%- elif mode == "REMOVE_COLUMN" %}
        min_missing_ratio = {{min_thresh}}
        max_missing_ratio = {{max_thresh}}
        cols = {{columns}}
        df_remove = ({{input}}.select(
            (pl.col(cols).is_null().sum() / pl.col(cols).count())
                .is_between(min_missing_ratio, max_missing_ratio)
                .suffix('_')).collect()[0])
        to_remove = [col for col in cols if df_remove[f'{col}_'][0]]
        if to_remove:
            {{output}} = {{input}}.select(pl.exclude(to_remove))
        else:
            {{output}} = {{input}}

    {%- elif mode == "VALUE" %}
        {{output}} = {{input}}.with_columns([
            {%- for col in columns %}
            pl.col('{{col}}').fill_null(value={{value}}),
            {%- endfor %}
        ])

    {%- elif mode in ("MEAN", "MEDIAN", "MODE") %}
        {{output}} = {{input}}.select([
            pl.exclude({{columns}}),
            {%- for col in columns %}
            pl.col('{{col}}').fill_null(
                value=pl.col('{{col}}').{{mode.lower()}}()),
            {%- endfor %}
        ])
    {%- else %}
        raise ValueError('{{invalid_mode}}'.format('{{mode}}'))
    {%- endif %}
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None
        ctx = {
            'multiplicity': self.parameters.get('multiplicity', {}).get(
                'input data', 1),
            'mode': self.mode,
            'min_thresh': self.min_ratio, 'max_thresh': self.max_ratio,
            'output': self.output,
            'input': self.named_inputs['input data'],
            'columns': self.attributes,
            'value': self.value,
            'invalid_mode': gettext('Invalid mode: {}'),
        }
        code = self.render_template(ctx)
        return dedent(code)

    @staticmethod
    def check_parameter(parameter):
        output = ""
        try:
            if parameter.isdigit():
                output = int(parameter)
            else:
                output = float(parameter)
        except ValueError:
            output = parameter

        return output


class DifferenceOperation(sk.DifferenceOperation):
    """
    Returns a new DataFrame containing rows in this frame but not in another
    frame.
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return

        in1 = self.named_inputs['input data 1']
        in2 = self.named_inputs['input data 2']
        error = ('For difference operation, both input data '
                 'sources must have the same number of attributes '
                 'and types.')

        code = f"""
        if len({in1}.columns) != len({in2}.columns):
            raise ValueError('{error}')
        {self.output} = {in1}.join(
            left_on={in1}.columns, right_on={in2}.columns, how='anti')
        """
        return dedent(code)


class DistinctOperation(sk.DistinctOperation):
    """
    Returns a new DataFrame containing the distinct rows in this DataFrame.
    Parameters: attributes to consider during operation (keys)
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        in1 = self.named_inputs['input data']
        columns = self.attributes

        # Ok
        code = f"{self.output} = {in1}.unique(subset={columns}, keep='first')"
        return dedent(code)


class DropOperation(sk.DropOperation):
    """
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    The only parameters is the name of the columns to be removed.
    Since 2.6
    """
    ATTRIBUTES_PARAM = 'attributes'

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        input = self.named_inputs['input data']
        columns = self.attributes
        # Ok
        code = f"{self.output} = {input}.drop(columns={columns})"
        return dedent(code)


class ExecutePythonOperation(sk.ExecutePythonOperation):

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        in1 = self.named_inputs.get('input data 1', 'None')
        in2 = self.named_inputs.get('input data 2', 'None')

        if self.plain:
            if self.parameters.get('export_notebook'):
                code = dedent("""
                    # Input data
                    in1 = {in1}
                    in2 = {in2}
                    # Output data, initialized as None
                    out1 = None
                    out2 = None
                    DataFrame = pd.DataFrame
                    createDataFrame = pd.DataFrame
                    """.format(in1=in1, in2=in2))
                code = code + '\n' + self.code
                return code
            else:
                return dedent(self.code)

        code = dedent("""

        results = [r[1].result() for r in task_futures.items() if r[1].done()]
        results = dict([(r['task_name'], r) for r in results])
        # Input data
        in1 = {in1}
        in2 = {in2}
        # Output data, initialized as None
        out1 = None
        out2 = None
        # Variables and language supportedn
        ctx = {{
            'wf_results': results,
            'in1': in1,
            'in2': in2,
            'out1': out1,
            'out2': out2,
            'DataFrame': pl.DataFrame,
            'createDataFrame': pl.DataFrame,
            'pl': pl,

            # Restrictions in Python language
            '_write_': lambda v: v,
            # '_getattr_': getattr,
            '_getitem_': lambda ob, index: ob[index],
            '_getiter_': lambda it: it,
            '_print_': PrintCollector,
            'json': json,
        }}
        user_code = {code}.decode('unicode_escape')

        ctx['__builtins__'] = safe_builtins

        compiled_code = compile_restricted(user_code,
            str('python_execute_{order}'), str('exec'))
        try:
            exec(compiled_code, ctx)

            # Retrieve values changed in the context
            out1 = ctx['out1']
            out2 = ctx['out2']
            if '_print' in ctx:
                emit_event(name='update task',
                    message=ctx['_print'](),
                    status='RUNNING',
                    identifier='{id}')
        except NameError as ne:
            raise ValueError(gettext('Invalid name: {{}}. '
                'Many Python commands are not available in Lemonade').format(ne))
        except ImportError as ie:
            raise ValueError(gettext('Command import is not supported'))
        """.format(in1=in1, in2=in2, code=self.code.encode('unicode_escape'),
                   order=self.order,
                   id=self.parameters['task']['id']))

        code += dedent("""
        {out1} = out1
        {out2} = out2
        """.format(out1=self.out1, out2=self.out2))
        return dedent(code)


class ExecuteSQLOperation(sk.ExecuteSQLOperation):

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

        raise ValueError(
            gettext('ExecuteSQLOperation not supported by this variant.'))


class FilterOperation(sk.FilterOperation):
    """
    Filters rows using the given condition.
    Parameters:
        - A boolean expression
    Since 2.6
    """
    template = """
        {%- if expressions %}
        {{out}} = {{input}}.filter(
        {%- for expr in expressions %}
            ({{expr}}){% if not loop.last %} & {% endif %}
        {%- endfor %}
        )
        {%- else %}
        {out} = {input}
        {%- endif %}
        """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if self.has_code:
            input_data = self.named_inputs['input data']
            params = {'input': input_data}

            expressions = []
            expression = Expression(None, params)
            if self.advanced_filter:
                for expr in self.advanced_filter:
                    expressions.append(expression.parse(expr['tree'], params))
                    self.transpiler_utils.add_import(expression.imports)
            else:
                raise ValueError(gettext('Parameter {} is required').format(
                    'expression'))

            ctx = {
                'out': self.output,
                'input': self.named_inputs['input data'],
                'expressions': expressions
            }
            return dedent(self.render_template(ctx))


class IntersectionOperation(sk.IntersectionOperation):
    """
    Returns a new DataFrame containing rows only in both this
    frame and another frame.
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        in1 = self.named_inputs['input data 1']
        in2 = self.named_inputs['input data 2']
        error = ('For intersection operation, both input data '
                 'sources must have the same number of attributes '
                 'and types.')

        code = f"""
        if len({in1}.columns) != len({in2}.columns):
            raise ValueError('{error}')
        {self.output} = {in1}.join(
            left_on={in1}.columns, right_on={in2}.columns, how='semi')
        """
        return dedent(code)


class JoinOperation(sk.JoinOperation):
    """
    Joins with another DataFrame, using the given join expression.
    The expression must be defined as a string parameter.
    Since 2.6
    """

    template = """
        {%- if prefix1 %}
        keys1 = ({% for k in keys1 %}'{{prefix1}}{{k}}', {% endfor %})
        {{in1}} = {{in1}}.select([
            pl.col(c).alias(f'{{prefix1}}{c}')
            for c in {{in1}}.columns
        ])
        {%- else %}
        keys1 = {{keys1}}
        {%- endif %}

        {%- if prefix2 %}
        keys2 = ({% for k in keys2 %}'{{prefix2}}{{k}}', {% endfor %})
        {{in2}} = {{in2}}.select([
            pl.col(c).alias(f'{{prefix2}}{c}')
            for c in {{in2}}.columns
        ])
        {%- else %}
        keys2 = {{keys2}}
        {%- endif %}

        {%- if match_case %}
        # Identify Utf8 keys in order to convert to lower case
        # and perform case-insensitive join.
        string_cols1 = {c for (i, c) in
            enumerate({{in1}}.columns)
            if {{in1}}.dtypes[i] == pl.Utf8 and c in keys1}
        string_cols2 = {c for (i, c) in
            enumerate({{in2}}.columns)
            if {{in2}}.dtypes[i] == pl.Utf8 and c in keys2}

        keys1 = [pl.col(k)
            if k not in string_cols1
            else pl.col(k).str.to_lowercase()
            for k in keys1]
        keys2 = [pl.col(k)
            if k not in string_cols2
            else pl.col(k).str.to_lowercase()
            for k in keys2]
        {%- endif %}

        # Perform the join
        with pl.StringCache():
            {%- if type == 'right' %}
            # Polars does not support 'right' join
            # See https://github.com/pola-rs/polars/issues/3934
            # Invert the order to use left outer join
            {{out}} = {{in2}}.join({{in1}}, right_on=key1, left_on=keys2,
                how='left')
            # Revert columns' order
            {{out}} = {{out}}.select(
                [pl.col(c) if c not in keys1
                        else pl.col(keys2[keys1.index(c)]).alias(c)
                    for c in {{in1}}.columns] +
                [pl.col(c) for c in {{in2}}.columns
                    {%- if not keep_right_keys %} if c not in keys2 {%- endif %}])
            {%- else %}
            {{out}} = {{in1}}.join(
                {{in2}}, left_on=keys1, right_on=keys2, how='{{type}}')


            # Select the resulting attributes
            select = []
            {%- if selection_type1 == 1 %}
            select += {{in1}}.columns
            {%- elif selection_type1 == 2 %}
            select += [pl.col(k).alias(a)
                for k, a in {{selected_attrs1}}
            ]
            {%- else %}
            # No selection from 1st dataset
            {%- endif %}

            {%- if selection_type2 == 1 %}
            select += [c for c in {{in2}}.columns if c not in keys2]
            {%- elif selection_type2 == 2 %}
            select += [pl.col(k).alias(a)
                for k, a in {{selected_attrs2}}
            ]
            {%- else %}
            # No selection from 2nd dataset
            {%- endif %}

            {{out}} = {{out}}.select(select)

            {% if keep_right_keys %}
            # Keep the right keys
            # {{out}} = {{out}}.select(
            #    [pl.col(c) for c in {{in1}}.columns] +
            #    [pl.col(c)
            #        if c not in keys2 else pl.col(keys1[keys2.index(c)]).alias(c)
            #        for c in {{in2}}.columns])
            {%- endif %}
            {%- endif %}
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def _get_operator(self, value: str) -> str:
        operators = {
            'ne': '!=',
            'gt': '>',
            'lt': '<',
            'ge': '>=',
            'le': '<='
        }
        return operators.get(value, '==')

    def generate_code(self):
        if not self.has_code:
            return None
        input_data1 = self.named_inputs['input data 1']
        input_data2 = self.named_inputs['input data 2']

        on_clause = [(f['first'], f['second'], self._get_operator(f.get('op', '=')))
                for f in self.join_parameters.get('conditions')]

        first_prefix = (self.join_parameters.get(
            'firstPrefix', 'first') or 'first').strip()
        second_prefix = (self.join_parameters.get(
            'secondPrefix', 'second') or 'second').strip()

        conditions = self.join_parameters.get('conditions')
        keys1, keys2 = zip(*[
            [x.get('first'), x.get('second')] for x in conditions])
        selection_type1 = self.join_parameters.get('firstSelectionType')
        selection_type2 = self.join_parameters.get('secondSelectionType')
        select1 = self.join_parameters.get('firstSelect')
        select2 = self.join_parameters.get('secondSelect')
        prefix1 = self.join_parameters.get('firstPrefix')
        prefix2 = self.join_parameters.get('secondPrefix')

        selected_attrs1 = tuple([[attr.get('attribute'), attr.get('alias')]
            for attr in (self.join_parameters.get('firstSelect', []) or [])
            if attr.get('select')
        ])
        selected_attrs2 = tuple([[attr.get('attribute'), attr.get('alias')]
            for attr in (self.join_parameters.get('secondSelect', []) or [])
            if attr.get('select')
        ])

        ctx = dict(
            out=self.output,
            in1=input_data1,
            in2=input_data2,
            pref_l=first_prefix, pref_r=second_prefix,
            keys1=keys1, keys2=keys2,
            match_case=self.match_case,
            type=self.join_type,
            selection_type1=selection_type1,
            selection_type2=selection_type2,
            select1=select1,
            select2=select2,
            prefix1=prefix1,
            prefix2=prefix2,
            selected_attrs1=selected_attrs1,
            selected_attrs2=selected_attrs2,
            keep_right_keys=not self.not_keep_right_keys)

        code = self.render_template(ctx)
        return dedent(code)


class RenameAttrOperation(sk.RenameAttrOperation):
    """Renames the attributes
    Since 2.6
    """
    ATTRIBUTES_PARAM = 'attributes'
    ALIAS_PARAM = 'alias'

    def __init__(self, parameters, named_inputs, named_outputs):
        sk.RenameAttrOperation.__init__(self, parameters, named_inputs,
                                        named_outputs)

    def generate_code(self):
        """Generate code."""
        if not self.has_code:
            return None

        rename = dict(zip(self.attributes, self.alias))
        return dedent(f"""
            {self.output} = {self.input}.rename({repr(rename)})
        """)


class ReplaceValuesOperation(sk.ReplaceValuesOperation):
    """
    Replace values in one or more attributes from a dataframe.
    Parameters:
    - The list of columns selected.
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)
        self.nullify = parameters.get('nullify', False) in ('1', 1, True)

    def generate_code(self):
        if not self.has_code:
            return None
        self.input = self.named_inputs['input data']
        self.template = """
            replacements = {{op.replaces}}
            to_select = []
            for col in {{op.input}}.columns:
                col_type = {{op.input}}.schema[col]
                if col in replacements:
                    replaces = replacements[col]
                    replacement = replaces[1][0]
                    value = replaces[0][0]
                    if col_type in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                        value = int(value)
                    elif col_type in (pl.Float32, pl.Float64):
                        value = float(value)
                    elif col_type in (pl.Utf8,):
                        value = str(value)
                    elif col_type in (pl.Boolean,):
                        value = bool(value)
                        if replacement in ('true', 'True', 1):
                            replacement = True
                        elif replacement in ('false', 'False', 0):
                            replacement = False

                    to_select.append(
                        pl.when(pl.col(col) == value)
                        .then(replacement)
                        .otherwise(pl.col(col)).alias(col))
                else:
                    to_select.append(pl.col(col))
            {{op.output}} = {{op.input}}.select(to_select)
            """
        return dedent(self.render_template({'op': self}))


class SampleOrPartitionOperation(sk.SampleOrPartitionOperation):
    """
    Returns a sampled subset of this DataFrame.
    Parameters:
    - fraction -> fraction of the data frame to be sampled.
        without replacement: probability that each element is chosen;
            fraction must be [0, 1]
    - seed -> seed for random operation.
    Since 2.6
    """

    template = """
        {%- if type == 'percent' %}
        {{output}} = {{input}}.collect().sample(
            frac={{fraction}}, shuffle=True, seed={{seed}}).lazy()
        {%- elif type == 'head' %}
        {{output}} = {{input}}.head({{value}})
        {%- else %}
        {{output}} = {{input}}.collect().sample(n={{value}},
            shuffle=True, seed={{seed}}).lazy()
    {%- endif %}
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        ctx = {
            'output': self.output,
            'input': self.named_inputs['input data'],
            'seed': self.seed,
            'value': self.value,
            'fraction': self.fraction,
            'type': self.type
        }
        code = self.render_template(ctx)
        return dedent(code)


class SelectOperation(sk.SelectOperation):
    """
    Projects a set of expressions and returns a new DataFrame.
    Parameters:
    - The list of columns selected.
    Since 2.6
    """
    template = """
        {%- if op.mode == 'exclude' %}

        exclude = {{op.attributes}}
        {{op.output}} = {{op.input}}.select(pl.all().exclude(exclude))

        {% elif op.mode in ('include', 'legacy') %}
        {{op.output}} = {{op.input}}.select([
            {%- for attr, alias in op.alias_tuple %}
            pl.col('{{attr}}')
                {%- if attr != alias%}.alias('{{alias}}'){% endif %},
            {%- endfor %}
        ])
        {%- elif op.mode == 'rename' %}
        {{op.output}} = {{op.input}}.rename(
            mapping=dict({{op.alias_tuple}}))

        {%- elif op.mode == 'duplicate' %}
        {{op.output}} = {{op.input}}.select([
            pl.all(),
            {%- for k, v in op.alias_tuple %}
            pl.col('{{k}}').alias('{{v}}'),
            {%- endfor %}
        ]).lazy()
        {%- endif %}
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return

        attributes = []
        aliases = []
        alias_tuple = []
        for attr in self.attributes:
            # legacy format, without alias
            if self.mode is None or self.mode == 'legacy':
                attributes.append(attr)
            else:
                attribute_name = attr.get('attribute')
                attributes.append(attribute_name)

                alias = attr.get('alias')
                aliases.append(alias or attribute_name)
                alias_tuple.append([attribute_name, alias or attribute_name])

        return dedent(self.render_template(
            {'op': {'attributes': attributes, 'aliases': aliases,
                    'mode': self.mode or 'include',
                    'input': self.named_inputs['input data'],
                    'output': self.output,
                    'alias_tuple': alias_tuple}}))


class SortOperation(sk.SortOperation):
    """
    Returns a new DataFrame sorted by the specified column(s).
    Parameters:
    - The list of columns to be sorted.
    - A list indicating whether the sort order is ascending for the columns.
    Condition: the list of columns should have the same size of the list of
               boolean to indicating if it is ascending sorting.
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None
        input = self.named_inputs['input data']
        reverse = [not x for x in self.ascending]

        code = f"""
            {self.output} = {input}.sort(
                by={repr(self.columns)}, descending={repr(reverse)}).lazy()
        """

        return dedent(code)


class SplitKFoldOperation(sk.SplitKFoldOperation):
    """
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    template = """
        {%- if op.stratified %}
        {{op.output}} = ({{op.named_inputs['input data']}}
            {%- if op.shuffle == 1 %}
            .collect()
            .sample(frac=1, shuffle=True, seed={{op.random_state}})
            {%- endif %}
            .with_column(pl.lit(1).alias('{{op.alias}}'))
            .select([pl.exclude('{{op.alias}}'),
                (pl.col('{{op.alias}}').cumsum().over('{{op.column}}')
                    % {{op.n_splits}}).alias('{{op.alias}}')]).lazy()
        )
        {%- else %}
        {{op.output}} = ({{op.named_inputs['input data']}}
            {%- if op.shuffle == 1 %}
            .collect()
            .sample(frac=1, shuffle=True, seed={{op.random_state}})
            {%- endif %}
            .with_row_count('{{op.alias}}')).lazy()

        k = {{op.n_splits}}
        size = {{op.output}}.select(pl.count()).collect().to_pandas().iloc[0][0]
        fold_size = size // k
        remainder = size %  k

        col = pl.col('{{op.alias}}')
        replaces = None
        for i in range(k):
            fold_start_index = i * fold_size
            if i < remainder:
                fold_start_index += i
                fold_end_index = fold_start_index + fold_size + 1
            else:
                fold_start_index += remainder
                fold_end_index = fold_start_index + fold_size
            if replaces == None:
                replaces = (pl.when(
                    (col >= fold_start_index) & (col < fold_end_index))
                    .then(pl.lit(i)))
            else:
                replaces = (replaces.when(
                    (col >= fold_start_index) & (col < fold_end_index))
                    .then(pl.lit(i)))

        {{op.output}} = ({{op.output}}
            .select([pl.exclude('{{op.alias}}'),
                replaces.alias('{{op.alias}}')]).lazy()
        )
        {%- endif %}
    """

    def generate_code(self):
        if not self.has_code:
            return None
        return dedent(self.render_template({'op': self}))


class SplitOperation(sk.SplitOperation):
    """
    Randomly splits a Data Frame into two data frames.
    Parameters:
    - List with two weights for the two new data frames.
    - Optional seed in case of deterministic random operation
        ('0' means no seed).
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        code = f"""
            shuffled = {self.named_inputs['input data']}.collect().sample(
                frac=1.0, seed={self.seed}, shuffle=True)
            percent = {repr(self.weights)}
            pos = round(shuffled.shape[0] * percent)

            {self.out1} = shuffled.slice(0, pos).lazy()
            {self.out2} = shuffled.slice(pos).lazy()
        """
        return dedent(code)


class TransformationOperation(sk.TransformationOperation):
    """
    Returns a new DataFrame applying the expression to the specified column.
    Parameters:
        - Alias: new column name. If the name is the same of an existing,
            replace it.
        - Expression: json describing the transformation expression
    Since 2.6
    """
    template = """
        {%- if functions %}
        {{out}} = {{input}}.with_columns([
            {%- for (alias, f) in functions %}
            {{f}}.alias('{{alias}}'),
            {%- endfor %}
        ]).lazy()
        {%- else %}
        {{out}} = {{input}}
        {%- endif %}

        {%- if use_positions %}
        # Reorder columns. Positions: {{positions}}
        new_columns = set([
            {%- for (alias, f) in functions -%}
            '{{alias}}', {% endfor%}])
        new_cols = [c for c in {{out}}.columns if c not in new_columns]
        {%- for pos in positions %}
        new_cols.insert({{pos}}, '{{functions[loop.index0][0]}}')
        {%- endfor %}
        {{out}} = {{out}}.select(new_cols).lazy()
        {%- endif %}
        """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)
        if self.has_code:
            # Builds the expression and identify the target column
            # Must be executed in constructor to correctly define
            # the list of necessary imports
            params = {'input': self.named_inputs.get('input data')}
            self.functions = []
            expression = Expression(None, None)

            for expr in self.expressions:
                f = expression.parse(expr['tree'], params)
                self.functions.append((expr['alias'], f))
                if expression.imports:
                    for imp in expression.imports:
                        self.transpiler_utils.add_import(imp)

                if expression.custom_functions:
                    for name, cf in expression.custom_functions.items():
                        self.transpiler_utils.add_custom_function(name, cf)

        self.parameters = parameters

    def generate_code(self):

        ctx = {
            'out': self.output, 'input': self.named_inputs['input data'],
            'functions': self.functions, 'positions': self.positions,
            'use_positions': bool([p for p in self.positions if p > -1])
        }
        code = self.render_template(ctx)
        return dedent(code)


class UnionOperation(sk.UnionOperation):
    """
    Return a new DataFrame containing all rows in this frame and another frame.
    Takes no parameters.
    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        df1 = self.named_inputs['input data 1']
        df2 = self.named_inputs['input data 2']

        return dedent(
            f"{self.output} = pl.concat([{df1}, {df2}], how='vertical').lazy()")
