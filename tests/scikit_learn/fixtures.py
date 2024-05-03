import pytest
from tests.scikit_learn.util import _read_data
import polars as pl
import importlib
from tests.scikit_learn import util

# @pytest.fixture(scope="session")
# def iris(request):
#     columns, size = request.param
#     return read('iris', columns, size)

@pytest.fixture(scope="session")
def iris():
    return _read_data('iris')

def wine(columns=None, size=None):
    return _read_data('wine', columns, size)


def titanic(columns=None, size=None):
    return _read_data('titanic', columns, size)


def _identity(x): return x


def get_parametrize(module_name: str, op_name: str):
    """Return parameters for multiple executions/tests, based on the different
    implementations/variants for the platform.

    Returns:
        dict: Dict containing parameters as defined in pytest.mark.parametrize()
    """
    def instantiate(variant: str):
        final_name = (f'juicer.scikit_learn.{variant}.{module_name}_operation'
            if variant else f'juicer.scikit_learn.{module_name}_operation')
        module = importlib.import_module(final_name)
        return getattr(module, op_name)

    return {
        'argnames': ("impl", "source", "target"),
        'argvalues': [
            (instantiate(None), _identity, _identity),
            # (OpDuck, None, lambda x: x.df()),
            (instantiate('polars'), lambda df: pl.from_pandas(df).lazy(),
                lambda x: x.collect().to_pandas(),)
        ],
        'ids': ['pandas',  # 'duckdb',
                'polars', ]
    }

@pytest.fixture(scope='session')
def get_df():
    return util.iris_polars()