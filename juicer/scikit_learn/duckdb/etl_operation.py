# -*- coding: utf-8 -*-
from gettext import gettext
from textwrap import dedent

from juicer.scikit_learn.duckdb.expression import Expression
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

        con = 'get_global_duckdb_conn()'
        # Ok
        code = f"""
            ok_columns = set({input1}.columns).symmetric_difference(
                set({input2}.columns))

            cols1 = []
            for col in {input1}.columns:
                    cols1.append(col if col in ok_columns else
                        f'{{col}} AS {s1}{{col}}')
 
            # Convert to pandas because Python API do not support FULL OUTER JOIN
            tmp1 = {input1}.project(
                f'{{", ".join(cols1)}}, row_number() over () AS _tmp_1')
            tmp1.create_view('tb1_{self.order}')

            cols2 = []
            for col in {input2}.columns:
                    cols2.append(col if col in ok_columns else
                        f'{{col}} AS {s1}{{col}}')

            tmp2 = {input2}.set_alias('{s2}').project(
                f'{{", ".join(cols2)}}, row_number() over () AS _tmp_2')
            tmp2.create_view('tb2_{self.order}')

            {self.output} = {con}.query('''
                SELECT * FROM tb1_{self.order} AS {s1}
                FULL OUTER JOIN tb2_{self.order} AS {s2} 
                    ON {s1}._tmp_1 = {s2}._tmp_2'''
                ).project('* EXCLUDE (_tmp_1, _tmp_2)')
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
    {%- if pivot %}
    raise ValueError('{{unsupported}}')
    {%- else %}
    {{output}} = {{input}}.aggregate(
        aggr_expr='''
        {%- for agg in aggregations %}
            {%- if agg.f == 'countDistinct' %}
            COUNT(DISTINCT {{agg.attribute}}) AS {{agg.alias}}
            {%- elif agg.f == 'sumDistinct' %}
            SUM(DISTINCT {{agg.attribute}}) AS {{agg.alias}}
            {%- else %}
            {{agg.f.upper()}}({{agg.attribute}}) AS {{agg.alias}}
            {%- endif %}
            {%- if not loop.last%}, {% endif %}
        {%- endfor %}
        ''',
        group_expr='{{columns|join(', ')}}'
    )
    {%- endif %}
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None
        ctx = {
            'columns': self.attributes,
            'pivot': self.pivot,
            'pivot_values': self.pivot_values,
            'input': self.named_inputs['input data'],
            'output': self.output,
            'agg_func': self.input_operations_pivot,
            'aggregations': self.functions,
            'unsupported': gettext('Pivot is not supported.')
        }
        code = self.render_template(ctx)
        return dedent(code)


class CastOperation(sk.CastOperation):
    """ Change attribute type.
    Since 2.6
    """

    template = """
        try:
            {{op.output}} = {{op.input}}.project('''
                * EXCLUDE(
                {%- for attr in op.attributes %}{{attr.attribute}}
                {%- if not loop.last %}, {% endif %}
                {%- endfor %}),
        {%- for attr in op.attributes %}
            {%- if attr.type == 'Integer' %}
                TRY_CAST({{attr.attribute}} AS INTEGER)
            {%- elif attr.type == 'Decimal' %}
                TRY_CAST({{attr.attribute}} AS FLOAT)
            {%- elif attr.type == 'Boolean' %}
                TRY_CAST({{attr.attribute}} AS BOOLEAN)
            {%- elif attr.type == 'Date' %}
                TRY_CAST({{attr.attribute}} AS DATE)
            {%- elif attr.type in ('DateTime', 'Datetime') %}
                TRY_CAST({{attr.attribute}} AS DATETIME)
            {%- elif attr.type in ('Time', ) %}
                TRY_CAST({{attr.attribute}} AS TIME)
            {%- elif attr.type == 'Text' %}
                CAST({{attr.attribute}} AS VARCHAR)
            {%- elif attr.type == 'Array' %}
                CAST({{attr.attribute}} AS LIST)
            {%- elif attr.type == 'JSON' %}
                TO_JSON({{attr.attribute}})
            {%-endif %} AS {{attr.attribute}}{% if not loop.last%}, {%endif%}
        {%- endfor %}
            ''')
        except ValueError as ve:
            msg = str(ve)
            if 'Unable to parse string' in msg:
                expr = re.compile(r'.+string "(.+)" at position (\\d+)')
                parts = expr.findall(msg)[0]
                raise ValueError('{{errors.unable_to_parse}}'.format(*parts))
            else:
                raise
    {%- if op.errors == 'move' %}
        {%- for attr in op.attributes %}
        # Copy invalid data to a new attribute
        # Invalid output rows have NaN in cells, but not in input.
        #s = ({{op.input}}['{{attr.attribute}}'].notnull() != {{op.output}}['{{attr.attribute}}'].notnull())
        #{{op.output}}.loc[s, '{{op.invalid_values}}'] = {{op.input}}['{{attr.attribute}}']
        {%- endfor %}
    {%- endif %}
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
        {{output}} = {{input}}.filter('''
            {%- for col in columns %}
            {{col}} IS NOT NULL {% if not loop.last %} AND {% endif %}
            {%- endfor %}
        ''')

    {%- elif mode == "REMOVE_COLUMN" %}
        cols = {{columns}}
        miss_ratio_df = {{input}}.aggregate('''
            {%- for col in columns %}
            1.0 * SUM(CASE WHEN {{col}} IS NULL THEN 1 ELSE 0 END) / COUNT(*) 
                BETWEEN {{min_thresh}} AND {{max_thresh}} AS _tmp_{{col}}
            {%- if not loop.last %}, {% endif %}
            {%- endfor %}
        ''').df()
        to_remove = [col for col in cols if miss_ratio_df[f'_tmp_{col}'][0]]
        if to_remove:
            {{output}} = {{input}}.project(f"* EXCLUDE ({','.join(to_remove)})")
        else:
            {{output}} = {{input}}

    {%- elif mode == "VALUE" %}
        {{output}} = {{input}}.project('''
            * EXCLUDE ({{columns|join(', ')}}), 
            {%- for col in columns %}
            COALESCE({{col}}, {{value}}) AS {{col}}
            {%- if not loop.last %},{% endif %}
            {%- endfor %}
        ''')

    {%- elif mode in ("MEAN", "MEDIAN", "MODE", "AVG") %}
        {{output}} = {{input}}.project('''
            * EXCLUDE ({{columns|join(', ')}}), 
            {%- for col in columns %}
            COALESCE({{col}}, {{mode.upper()}}({{col}}) OVER ()) AS {{col}}
            {%- endfor %}
        ''')
    {%- else %}
        raise ValueError('{{invalid_mode}}'.format('{{mode}}'))
    {%- endif %}
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None
        copy_code = ".copy()" \
            if self.parameters.get('multiplicity',
                                   {}).get('input data', 1) > 1 else ""
        value_is_str = isinstance(self.check_parameter(self.value), str)
        if value_is_str:
            self.value = f"'{self.value}'"
        ctx = {
            'multiplicity': self.parameters.get('multiplicity', {}).get(
                'input data', 1),
            'mode': self.mode if self.mode not in ('mean', 'MEAN') else 'AVG',
            'min_thresh': self.min_ratio, 'max_thresh': self.max_ratio,
            'copy_code': copy_code, 'output': self.output,
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
        {self.output} = {in1}.except_({in2})
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
        cols = self.attributes

        # Ok
        return f"{self.output} = {in1}.project('{', '.join(cols)}').distinct()"


class DropOperation(sk.DropOperation):
    """
    Returns a new DataFrame that drops the specified column.
    Nothing is done if schema doesn't contain the given column name(s).
    The only parameters is the name of the columns to be removed.

    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        input = self.named_inputs['input data']
        code = f"""{self.output} = {input}.project(
            '* EXCLUDE({', '.join(self.attributes)})')"""
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
            'in1': in1.df() if in1 is not None else None,
            'in2': in2.df() if in2 is not None else None,,
            'out1': out1,
            'out2': out2,
            'DataFrame': pl.DataFrame,
            'createDataFrame': pd.DataFrame,

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
            df_out1 = ctx['out1']
            df_out2 = ctx['out2']
            
            out1 = (con.execute('SELECT * FRM df_out1') if df_out1 is not None
                else None)
            out2 = (con.execute('SELECT * FRM df_out2') if df_out2 is not None
                else None)
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
    """
    Execute SQL.
    Since 2.6
    """
    template = """
        {%- if op.input1 %}
        con.register('ds1', {{op.input1}}.df())
        {%- endif %}
        {%- if op.input2 %}
        con.register('ds2', {{op.input2}}.df())
        {%- endif %}
        {{op.output}} = con.query(\"\"\"
            {%- for line in op.query.split('\n') %}
            {{line}}
            {%- endfor %}
        \"\"\")
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        return dedent(self.render_template({'op': self}))


class FilterOperation(sk.FilterOperation):
    """
    Filters rows using the given condition.
    Parameters:
        - A boolean expression

    Since 2.6
    """
    template = """
        {%- if expressions %}
        {{out}} = {{input}}.filter('''
        {%- for expr in expressions %}
            {{expr}}{% if not loop.last %} AND {% endif %}
        {%- endfor %}
        ''')
        {%- else %}
        {out} = {input}
        {%- endif %}
        """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        input_data = self.named_inputs['input data']
        params = {'input': input_data}

        expressions = []
        expression = Expression(None, params)
        for expr in self.advanced_filter:
            expressions.append(expression.parse(expr['tree'], params))
            self.transpiler_utils.add_import(expression.imports)

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
        {self.output} = {in1}.intersect({in2})
        """
        return dedent(code)


class JoinOperation(sk.JoinOperation):
    """
    Joins with another DataFrame, using the given join expression.
    The expression must be defined as a string parameter.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        self.template = """

        """
        code = """
        cols1 = [ c + '{suf_l}' for c in {in1}.columns]
        cols2 = [ c + '{suf_r}' for c in {in2}.columns]

        {in1}.columns = cols1
        {in2}.columns = cols2

        keys1 = [c + '{suf_l}' for c in {keys1}]
        keys2 = [c + '{suf_r}' for c in {keys2}]
        """.format(in1=self.named_inputs['input data 1'],
                   in2=self.named_inputs['input data 2'],
                   suf_l=self.suffixes[0], suf_r=self.suffixes[1],
                   keys1=self.left_attributes, keys2=self.right_attributes)

        # Should be positive boolean logic? ---> '''if self.match_case:'''
        if not self.match_case:
            code += """
        data1_tmp = {in1}[keys1].applymap(lambda col: str(col).lower()).copy()
        data1_tmp.columns = [c + "_lower" for c in data1_tmp.columns]
        col1 = list(data1_tmp.columns)
        data1_tmp = pd.concat([{in1}, data1_tmp], axis=1, sort=False)

        data2_tmp = {in2}[keys2].applymap(lambda col: str(col).lower()).copy()
        data2_tmp.columns = [c + "_lower" for c in data2_tmp.columns]
        col2 = list(data2_tmp.columns)
        data2_tmp = pd.concat([{in2}, data2_tmp], axis=1, sort=False)

        {out} = pd.merge(data1_tmp, data2_tmp, left_on=col1, right_on=col2,
            copy=False, suffixes={suffixes}, how='{type}')
        # Why drop col_lower?
        {out}.drop(col1+col2, axis=1, inplace=True)
                """.format(out=self.output, type=self.join_type,
                           in1=self.named_inputs['input data 1'],
                           in2=self.named_inputs['input data 2'],
                           suffixes=self.suffixes)
        else:
            code += """
        {out} = pd.merge({in1}, {in2}, how='{type}',
                suffixes={suffixes},
                left_on=keys1, right_on=keys2).copy()
                """.format(out=self.output, type=self.join_type,
                           in1=self.named_inputs['input data 1'],
                           in2=self.named_inputs['input data 2'],
                           suffixes=self.suffixes)

        if self.not_keep_right_keys:
            code += """
        cols_to_remove = keys2
        {out}.drop(cols_to_remove, axis=1, inplace=True)
            """.format(out=self.output)

        return dedent(code)


class RenameAttrOperation(sk.RenameAttrOperation):
    """Renames the attributes

    Since 2.6
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        sk.RenameAttrOperation.__init__(self, parameters, named_inputs,
                                        named_outputs)

    def generate_code(self):
        """Generate code."""
        if not self.has_code:
            return None

        rename = [f'{attr} AS {alias}'
                  for attr, alias in zip(self.attributes, self.alias)]
        code = f"""
            {self.output} = {self.input}.project('''
                * EXCLUDE ({', '.join(self.attributes)}),
                {', '.join(rename)}
            ''')
        """
        return dedent(code)


class ReplaceValuesOperation(sk.ReplaceValuesOperation):
    """
    Replace values in one or more attributes from a dataframe.
    Parameters:
    - The list of columns selected.

    Since 2.6
    """
    template = """
        {{op.output}} = {{op.input}}.project('''
            * EXCLUDE ({{op.replaces.keys()|join(', ')}}),
        {%- for attr, pairs in op.replaces.items() %}
            CASE {{attr}}
            {%- for value in pairs[0] %}
            {%- if not loop.last %}, {%- endif %}
                WHEN '{{value}}' THEN '{{pairs[1][loop.index0]}}'
            {%- endfor %}
            ELSE 
                {{attr}}
            END AS {{attr}}
            {%- if not loop.last %}, {%- endif %}
        {% endfor %}
        ''')
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)
        self.input = self.named_inputs['input data']

    def generate_code(self):
        if not self.has_code:
            return None
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
        {%- if op.type in ('percent', 'value') %}
        cols = ', '.join([f'{c} {t}' 
            for c, t in zip({{op.input}}.columns, {{op.input}}.dtypes)])
        {{op.input}}.create_view('_tmp_{{op.order}}')
       
        {{op.output}} = con.query('''
            SELECT * FROM _tmp_{{op.order}}
            USING SAMPLE
            {%- if op.type == 'percent' %}
                {{100*op.fraction}} PERCENT (BERNOULLI, {{op.seed}})
            {%- else %}
                RESERVOIR({{op.value}} ROWS) 
                    REPEATABLE ({{op.seed}})
            {% endif %} 
        ''')
        {%- elif op.type == 'head' %}
        {{op.output}} = {{op.input}}.limit({{op.value}})
    {%- endif %}
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)
        self.input = self.named_inputs['input data']

    def generate_code(self):
        if not self.has_code:
            return None
        return dedent(self.render_template({'op': self}))


class SelectOperation(sk.SelectOperation):
    """
    Projects a set of expressions and returns a new DataFrame.
    Parameters:
    - The list of columns selected.

    Since 2.6
    """
    template = """
        {%- if op.mode == 'exclude' %}
        
        {{op.output}} = {{op.input}}.project(
            '* EXCLUDE ({{op.attributes |join(', ')}})')

        {% elif op.mode in ('include', 'legacy') %}
          {%- if op.aliases %}
        {{op.output}} = {{op.input}}.project('
            {%- for attr, alias in op.alias_dict.items() -%}
            {{attr}}{% if attr != alias %} AS {{alias}}{% endif %}
            {%- if not loop.last%}, {% endif %}
            {%- endfor -%}
        ')
          {%- else %}
        {{op.output}} = {{op.input}}.project(
            '{{op.attributes |join(', ')}}'
        )
        {{op.output}}.columns = {{op.aliases}}
          {%- endif %}

        {%- elif op.mode == 'rename' %}
        aliases = {{op.alias_dict}}
        columns = [f'{col} AS {aliases.get(col, col)}' for col in {{op.input}}.columns]
        {{op.output}} = {{op.input}}.project(', '.join(columns))

        {%- elif op.mode == 'duplicate' %}
        {{op.output}} = {{op.input}}.project('''
            *, 
            {%- for k, v in op.alias_dict.items() %}
            {{k}} AS {{v}}{% if not loop.last %},{% endif %}
            {%- endfor %}
        ''')
        {%- endif %}
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        return super().generate_code()


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
        direction = ['asc' if x else 'desc' for x in self.ascending]

        # Ok
        order = ', '.join(
            [f'{c} {direction}'
                for c, direction in zip(self.columns, direction)])
        code = f"""
            {self.output} = {input}.order(
                '{order}'
            )
        """

        return dedent(code)


class SplitKFoldOperation(sk.SplitKFoldOperation):

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    template = """
        {%- if op.stratified %}
        # n = {{op.input}}.project('COUNT(DISTINCT {{op.column}})').df().loc[0][0]
        {%- if op.shuffle %}
        con.execute('SELECT setseed({{op.random_state}})')
        {%- endif %}
        {{op.output}} = {{op.input}}
            {%- if op.shuffle %}.order('random()'){% endif -%}
            .project('''
            *,
            NTILE({{op.n_splits}}) OVER (PARTITION BY {{op.column}} 
                ORDER BY {{op.column}}) AS {{op.alias}}
        ''')
        {%- else %}
        {% endif %}
        """

    def generate_code(self):
        if not self.has_code:
            return None

        self.input = self.named_inputs['input data']
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
        self.input = self.named_inputs['input data']

        code = f"""
            con.execute('select setseed(0.{self.seed})')
            shuffled = ({self.input}
                .project('*, RANDOM() AS _tmp_')
                .order('_tmp_')
                .project('* EXCLUDE(_tmp_)'))
            percent = {repr(self.weights)}
            total = shuffled.count('*').df().loc[0][0]
            pos = round(total * percent)

            {self.out1} = shuffled.limit(pos)
            {self.out2} = shuffled.limit(total - pos, pos)
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
        if not self.has_code:
            return None

        self.template = """
            {%- if functions %}
            {{out}} = {{input}}.project('''
                *, 
                {%- for (alias, f) in functions %}
                {{f}} AS {{alias}}{% if not loop.last %},{% endif %}
                {%- endfor %}
            ''')
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
            {{out}} = {{out}}.project(', '.join(new_cols))
            {%- endif %}
        """
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
        # Ok
        code = f"{self.output} = {df1}.union({df2}.project('*'))"
        return dedent(code)
