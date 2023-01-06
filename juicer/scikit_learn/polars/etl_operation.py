# -*- coding: utf-8 -*-
import re
from gettext import gettext
from textwrap import dedent

import polars as pl

from juicer.operation import Operation
from juicer.scikit_learn.polars.expression import (
    JAVA_2_PYTHON_DATE_FORMAT, Expression)
import juicer.scikit_learn.etl_operation as sk


class AddColumnsOperation(sk.AddColumnsOperation):
    """
    Merge two data frames, column-wise, similar to the command paste in Linux.
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
            {self.output} = pl.concat(
                [{input1}.select(pl.all().prefix('{s1}')).collect(), 
                    {input2}.select(pl.all().prefix('{s2}')).collect()], 
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
        collect_set, count, first, last, max, min, sum and size
    """

    template = """
    {%- if pivot %}:
        {%- if pivot_values %}:
    input_data = {input}.loc[{input}['{pivot}'].isin([{values}])]
        {%- else %}
    input_data = {input}
        {%-- endif %}
    {output} = pd.pivot_table(input_data, index={index},
            columns={pivot}, aggfunc={agg_func})
    # rename columns and convert to DataFrame
    {output}.reset_index(inplace=True)
    new_idx = [n[0] if n[1] == '' else "%s_%s_%s" % (n[0],n[1], n[2])
                    for n in {output}.columns.ravel()]
    {output}.columns = new_idx
    {%- else %}:
        {output} = {input}.groupby({columns}).agg({operations}).reset_index()
    {%- endif %}
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        ctx = {
            'pivot': self.pivot[0],
            'pivot_values': self.pivot_values,
            'input': self.named_inputs['input data'],
            'agg_func': self.input_operations_pivot,
            'operations': ', '.join(self.input_operations_non_pivot),
        }
        code = self.render_template(ctx)
        return dedent(code)


class CastOperation(sk.CastOperation):
    """ Change attribute type.
    """

    template = """
        try:
            {{op.output}} = {{op.input}}.select([
                pl.exclude(
                {%- for attr in op.attributes %}'{{attr.attribute}}',
                {%- endfor %}),
        {%- for attr in op.attributes %}
            {%- if attr.type == 'Integer' %}
                pl.col('{{attr.attribute}}').cast(pl.Int64, strict=False)
            {%- elif attr.type == 'Decimal' %}
                pl.col('{{attr.attribute}}').cast(pl.Float64, strict=False)
            {%- elif attr.type == 'Boolean' %}
                pl.col('{{attr.attribute}}').cast(pl.Boolean, strict=False)
            {%- elif attr.type == 'Date' %}
                pl.col('{{attr.attribute}}').cast(pl.Date, strict=False)
            {%- elif attr.type in ('DateTime', 'Datetime') %}
                pl.col('{{attr.attribute}}').cast(pl.Datetime, strict=False)
            {%- elif attr.type in ('Time', ) %}
                pl.col('{{attr.attribute}}').cast(pl.Time, strict=False)
            {%- elif attr.type == 'Text' %}
                pl.col('{{attr.attribute}}').cast(pl.Utf8)
            {%- elif attr.type == 'Array' %}
                pl.col('{{attr.attribute}}').apply(lambda x: [x])
            {%- elif attr.type == 'JSON' %}
                pl.col('{{attr.attribute}}').apply(to_json)
            {%-endif %}
        {%- endfor %}
            ])
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
        {{output}} = {{input}}.select([
            pl.exclude({{columns}}), 
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
        copy_code = ".copy()" \
            if self.parameters.get('multiplicity',
                                   {}).get('input data', 1) > 1 else ""
        value_is_str = isinstance(self.check_parameter(self.value), str)
        if value_is_str:
            self.value = f"'{self.value}'"
        ctx = {
            'multiplicity': self.parameters.get('multiplicity', {}).get(
                'input data', 1),
            'mode': self.mode,
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
                   name="execute_python", order=self.order,
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
                           id1=self.left_attributes,
                           id2=self.right_attributes,
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
        code = f"""
            {self.output} = {self.input}.rename({repr(rename)})
        """
        return dedent(code)


class ReplaceValuesOperation(sk.ReplaceValuesOperation):
    """
    Replace values in one or more attributes from a dataframe.
    Parameters:
    - The list of columns selected.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if self.has_code:
            code = """
            {out} = {in1}
            replacement = {replaces}
            for col in replacement:
                list_replaces = replacement[col]
                {out}[col] = {out}[col].replace(list_replaces[0], list_replaces[1])
            """.format(out=self.output, in1=self.named_inputs['input data'],
                       replaces=self.replaces)
            return dedent(code)


class SampleOrPartitionOperation(sk.SampleOrPartitionOperation):
    """
    Returns a sampled subset of this DataFrame.
    Parameters:
    - fraction -> fraction of the data frame to be sampled.
        without replacement: probability that each element is chosen;
            fraction must be [0, 1]
    - seed -> seed for random operation.
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
    """
    template = """
        {%- if op.mode == 'exclude' %}
        
        exclude = {{op.attributes}}
        {{op.output}} = {{op.input}}.select(pl.all().exclude(exclude))

        {% elif op.mode == 'include' %}
        selection = {{op.attributes}}
        {{op.output}} = {{op.input}}.select(selection)
          {%- if op.aliases %}
        {{op.output}}.columns = {{op.aliases}}
          {%- endif %}

        {%- elif op.mode == 'rename' %}
        {{op.output}} = {{op.input}}.rename(mapping={{op.alias_dict}})

        {%- elif op.mode == 'duplicate' %}
        {{op.output}} = {{op.input}}.select([
            pl.all(),
            {%- for k, v in op.alias_dict.items() %}
            pl.col('{{k}}').alias('{{v}}'),
            {%- endfor %}
        ])
        {%- endif %}
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return

        attributes = []
        aliases = []
        alias_dict = {}
        for attr in self.attributes:
            if self.mode is None:  # legacy format, without alias
                attributes.append(attr)
            else:
                attribute_name = attr.get('attribute')
                attributes.append(attribute_name)

                alias = attr.get('alias')
                aliases.append(alias or attribute_name)
                alias_dict[attribute_name] = alias or attribute_name

        return dedent(self.render_template(
            {'op': {'attributes': attributes, 'aliases': aliases,
                    'mode': self.mode or 'include',
                    'input': self.named_inputs['input data'],
                    'output': self.output,
                    'alias_dict': alias_dict}}))


class SortOperation(sk.SortOperation):
    """
    Returns a new DataFrame sorted by the specified column(s).
    Parameters:
    - The list of columns to be sorted.
    - A list indicating whether the sort order is ascending for the columns.
    Condition: the list of columns should have the same size of the list of
               boolean to indicating if it is ascending sorting.
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None
        input = self.named_inputs['input data']
        reverse = [not x for x in self.ascending]

        # Ok
        code = f"""
            {self.output} = {input}.sort(
                by={repr(self.columns)}, reverse={repr(reverse)})
        """

        return dedent(code)


class SplitKFoldOperation(sk.SplitKFoldOperation):

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if self.has_code:
            code = """"""
            copy_code = ".copy()" \
                if self.parameters['multiplicity']['input data'] > 1 else ""

            if self.stratified:
                code = """
        skf = StratifiedKFold(n_splits={n_splits}, shuffle={shuffle},
        random_state={random_state})

        {output_data} = {input}{copy_code}
        tmp = np.full(len({input}), fill_value=-1, dtype=int)
        j = 0
        y = {input}['{column}'].to_numpy().tolist()
        for _, test_index in skf.split({input}, y):
            tmp[test_index] = j
            j += 1
        {output_data}['{alias}'] = tmp
                    """.format(output=self.output,
                               copy_code=copy_code,
                               input=self.named_inputs['input data'],
                               n_splits=self.n_splits,
                               shuffle=self.shuffle,
                               random_state=self.random_state,
                               output_data=self.output,
                               column=self.column,
                               alias=self.alias)
                return dedent(code)
            else:
                code += """
        kf = KFold(n_splits={n_splits}, shuffle={shuffle},
        random_state={random_state})

        {output_data} = {input}{copy_code}
        tmp = np.full(len({input}), fill_value=-1, dtype=int)
        j = 0
        for _, test_index in kf.split({input}):
            tmp[test_index] = j
            j += 1
        {output_data}['{alias}'] = tmp
                    """.format(output=self.output,
                               copy_code=copy_code,
                               input=self.named_inputs['input data'],
                               n_splits=self.n_splits,
                               shuffle=self.shuffle,
                               random_state=self.random_state,
                               output_data=self.output,
                               alias=self.alias)
                return dedent(code)


class SplitOperation(sk.SplitOperation):
    """
    Randomly splits a Data Frame into two data frames.
    Parameters:
    - List with two weights for the two new data frames.
    - Optional seed in case of deterministic random operation
        ('0' means no seed).

    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        # Ok
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
        self.template = """
            {%- if functions %}
            {{out}} = {{input}}.with_columns([
                {%- for (alias, f) in functions %}
                {{f}}.alias('{{alias}}'),
                {%- endfor %}
            ])
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
            {{out}} = {{out}}.select(new_cols)
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
    """

    def __init__(self, parameters, named_inputs, named_outputs):
        super().__init__(parameters, named_inputs, named_outputs)

    def generate_code(self):
        if not self.has_code:
            return None

        df1 = self.named_inputs['input data 1']
        df2 = self.named_inputs['input data 2']
        # Ok
        code = f"{self.output} = pl.concat([{df1}, {df2}], how='vertical')"
        return dedent(code)
