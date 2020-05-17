import pytest
import pandas as pd
from tests.scikit_learn.util import read

@pytest.fixture(scope="session")
def iris(request):
    columns, size = request.param
    return read('iris', columns, size)

def wine(columns=None, size=None):
    return read('wine', columns, size)

def titanic(columns=None, size=None):
    return read('titanic', columns, size);
