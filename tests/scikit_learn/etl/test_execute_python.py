from tests.scikit_learn import util
from juicer.scikit_learn.etl_operation import ExecutePythonOperation
from textwrap import dedent
import pytest


class TestingBypass:
    # task_futures for testing
    @staticmethod
    def result():
        return None

    @staticmethod
    def done():
        return False


def _emit_event(output):
    # emit_event for testing
    def f(**kwargs):
        output.append(kwargs)
    return f


# ExecutePython
#
def test_execute_python_data_types_success():
    # 'range()' is allowed, it can cause problems...
    arguments = {
        'parameters': {'code': dedent("""
        str = 'tst'
        int = 1
        float = 0.1
        complex_number = 1j
        list = ['t', 'e', 's', 't']
        tuple = ('t', 'e', 's', 't')
        range = range(6)
        dict = {'tst': 'success'}
        set = {'t', 'e', 's', 't'}
        frozenset= ({'t', 'e', 's', 't'})
        bool = True
        bytes = b"Hello"
        """),
                       'task': {'id': 0}},
        'named_inputs': {
            'input data 1': None,
            'input data 2': None
        },
        'named_outputs': {
            'output data 1': 'out1',
            'output data 2': 'out2'
        }
    }

    output = []
    instance = ExecutePythonOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'task_futures': {'items': TestingBypass},
                           'emit_event': _emit_event(output)})

    assert result['user_code'] == dedent("""
        str = 'tst'
        int = 1
        float = 0.1
        complex_number = 1j
        list = ['t', 'e', 's', 't']
        tuple = ('t', 'e', 's', 't')
        range = range(6)
        dict = {'tst': 'success'}
        set = {'t', 'e', 's', 't'}
        frozenset= ({'t', 'e', 's', 't'})
        bool = True
        bytes = b"Hello"
    """)


def test_execute_python_prohibited_data_types_success():
    # 'byte_array' and 'memory_view' are prohibited
    arguments = {
        'parameters': {'code': dedent("""
        byte_array = bytearray(5)
        memory_view = memoryview(bytes(5))
        """),
                       'task': {'id': 0}},
        'named_inputs': {
            'input data 1': None,
            'input data 2': None
        },
        'named_outputs': {
            'output data 1': 'out1',
            'output data 2': 'out2'
        }
    }

    output = []
    with pytest.raises(ValueError) as val_err:
        instance = ExecutePythonOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'task_futures': {'items': TestingBypass},
                               'emit_event': _emit_event(output)})
    assert "name \'bytearray\' is not defined." \
           " Many Python commands are not available in Lemonade" in str(val_err)


def test_execute_python_keywords_success():
    # 'for' and 'while' are allowed, they can cause problems...
    arguments = {
        'parameters': {'code': dedent("""
        pass
        a = True and False
        assert not a
        def testing():
            global b
            b = 'hello'
            return a
        testing()
        del testing
        try:
            0/0
        except(ZeroDivisionError):
            pass
        finally:
            pass
        for i in range(10):
            if i == 5:
                continue
            if i == 8:
                break
        c = lambda x: x + 10
        c(10)
        d = 0
        e = []
        while d not in e:
            e.append(d)
        f = True or False
        assert f
        def creategen():
            lis = range(3)
            for i in lis:
                yield i*i
        creategen()
        """),
                       'task': {'id': 0}},
        'named_inputs': {
            'input data 1': None,
            'input data 2': None
        },
        'named_outputs': {
            'output data 1': 'out1',
            'output data 2': 'out2'
        }
    }

    output = []
    instance = ExecutePythonOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'task_futures': {'items': TestingBypass},
                           'emit_event': _emit_event(output)})
    assert result['user_code'] == dedent("""
        pass
        a = True and False
        assert not a
        def testing():
            global b
            b = 'hello'
            return a
        testing()
        del testing
        try:
            0/0
        except(ZeroDivisionError):
            pass
        finally:
            pass
        for i in range(10):
            if i == 5:
                continue
            if i == 8:
                break
        c = lambda x: x + 10
        c(10)
        d = 0
        e = []
        while d not in e:
            e.append(d)
        f = True or False
        assert f
        def creategen():
            lis = range(3)
            for i in lis:
                yield i*i
        creategen()
    """)


def test_execute_prohibited_python_keywords_success():
    # 'class', 'nonlocal', 'import', 'from' and 'as' are prohibited
    arguments = {
        'parameters': {'code': dedent("""
        from math import inf
        class FailClass:
            def failfunc():
                nonlocal x
                x = 10
        """),
                       'task': {'id': 0}},
        'named_inputs': {
            'input data 1': None,
            'input data 2': None
        },
        'named_outputs': {
            'output data 1': 'out1',
            'output data 2': 'out2'
        }
    }
    output = []
    with pytest.raises(SyntaxError) as stx_err:
        instance = ExecutePythonOperation(**arguments)
        result = util.execute(instance.generate_code(),
                              {'task_futures': {'items': TestingBypass},
                               'emit_event': _emit_event(output)})
    assert "Nonlocal statements are not allowed." in str(stx_err)


def test_execute_python_operators_success():
    # Assignment Operators are prohibited (e.g., +=, -=, *=, /=)
    arguments = {
        'parameters': {'code': dedent("""
        1 + 1
        1 - 1
        1 * 1
        1 / 1
        1 & 1
        1 ** 1
        1 // 1
        1 == 1
        1 != 1
        1 > 1
        1 < 1
        1 >= 1
        1 <= 1
        1 & 1
        1 | 1
        1 ^ 1
        1 << 1
        1 >> 1
        """),
                       'task': {'id': 0}},
        'named_inputs': {
            'input data 1': None,
            'input data 2': None
        },
        'named_outputs': {
            'output data 1': 'out1',
            'output data 2': 'out2'
        }
    }

    output = []
    instance = ExecutePythonOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'task_futures': {'items': TestingBypass},
                           'emit_event': _emit_event(output)})
    assert result['user_code'] == dedent("""
        1 + 1
        1 - 1
        1 * 1
        1 / 1
        1 & 1
        1 ** 1
        1 // 1
        1 == 1
        1 != 1
        1 > 1
        1 < 1
        1 >= 1
        1 <= 1
        1 & 1
        1 | 1
        1 ^ 1
        1 << 1
        1 >> 1
    """)


def test_execute_python_print_success():
    arguments = {
        'parameters': {'code': dedent("""
        x = 'hello_world'
        print(x)
        """),
                       'task': {'id': 0}},
        'named_inputs': {
            'input data 1': None,
            'input data 2': None
        },
        'named_outputs': {
            'output data 1': 'out1',
            'output data 2': 'out2'
        }
    }

    output = []
    instance = ExecutePythonOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'task_futures': {'items': TestingBypass},
                           'emit_event': _emit_event(output)})

    assert output[0]['message'] == 'hello_world\n'


def test_execute_python_dangerous_zfill_method_success():
    # The zfill() can cause a crash
    arguments = {
        'parameters': {'code': dedent("""
        str_ing = ''
        str_ing = str_ing.zfill(100)
        
        # Example on how it can crash/overflow
        # str_ing = str_ing.zfill(10000000000)
        
        print(str_ing)
        """),
                       'task': {'id': 0}},
        'named_inputs': {
            'input data 1': None,
            'input data 2': None
        },
        'named_outputs': {
            'output data 1': 'out1',
            'output data 2': 'out2'
        }
    }

    output = []
    instance = ExecutePythonOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'task_futures': {'items': TestingBypass},
                           'emit_event': _emit_event(output)})

    assert output[0]['message'] == ''.zfill(100) + '\n'


def test_execute_big_or_infinite_loops_success():
    # The user can create big or infinite loops
    arguments = {
        'parameters': {'code': dedent("""
        # Example 1:
        # for i in range(100000000000000000):
        #     pass
        
        # Example 2:
        # while True:
        #     pass
        """),
                       'task': {'id': 0}},
        'named_inputs': {
            'input data 1': None,
            'input data 2': None
        },
        'named_outputs': {
            'output data 1': 'out1',
            'output data 2': 'out2'
        }
    }

    output = []
    instance = ExecutePythonOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'task_futures': {'items': TestingBypass},
                           'emit_event': _emit_event(output)})


def test_execute_python_pandas_success():
    # the user can use pretty much every method from pandas, this may cause
    # problems because of the quantity of methods and future methods that will
    # be added
    slice_size = 10
    df1 = ['df1', util.iris(['class', 'petalwidth'], slice_size)]

    df2 = ['df2', util.iris(['class'], slice_size)]
    arguments = {
        'parameters': {'code': dedent("""
        out1 = in1.drop(columns=['class'])
        """),
                       'task': {'id': 0}},
        'named_inputs': {
            'input data 1': df1[0],
            'input data 2': df2[0]
        },
        'named_outputs': {
            'output data 1': 'out1',
            'output data 2': 'out2'
        }
    }

    output = []
    instance = ExecutePythonOperation(**arguments)
    result = util.execute(instance.generate_code(),
                          {'task_futures': {'items': TestingBypass},
                           'df1': df1[1],
                           'df2': df2[1],
                           'emit_event': _emit_event(output)})
    assert result['out1'].equals(df1[1].drop(columns=['class']))
