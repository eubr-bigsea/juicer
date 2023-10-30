import json
import pathlib
import pytest

from juicer.meta.transpiler import MetaTranspiler

config = {
    'juicer': {
        'auditing': False,
        'services': {
            'limonero': {
                'url': 'http://localhost',
                'auth_token': 22222222
            }
        }

    }
}
@pytest.fixture(scope='session')
def transpiler():
    return MetaTranspiler(config)


@pytest.fixture(scope='session')
def titanic_workflow():
    module_dir = pathlib.Path(__file__).resolve().parent

    with open(module_dir / 'workflow.json') as f:
        return json.load(f)


def mock_get_datasource(*args):
    module_dir = pathlib.Path(__file__).resolve().parent.parent
    titanic = module_dir / 'data' / 'titanic.csv.gz'
    return {
        'storage': {'id': 100, 'name': 'Local'},
        'format': 'CSV',
        'url': f'file://{titanic}',
        'is_first_line_header': True,
        'infer_schema': 'FROM_LIMONERO',
        'attributes': [
            {'name': 'seq', 'type': 'INTEER'},
            {'name': 'pclass', 'type': 'CHARACTER'},
            {'name': 'survived', 'type': 'INTEGER'},
            {'name': 'name', 'type': 'CHARACTER'},
            {'name': 'sex', 'type': 'CHARACTER'},
            {'name': 'age', 'type': 'CHARACTER'},
            {'name': 'sibsp', 'type': 'ITNEGER'},
            {'name': 'parch', 'type': 'INTEGER'},
            {'name': 'ticket', 'type': 'CHARACTER'},
            {'name': 'fare', 'type': 'DECIMAL'},
            {'name': 'cabin', 'type': 'CHARACTER'},
            {'name': 'embarked', 'type': 'CHARACTER'},
            {'name': 'boat', 'type': 'CHARACTER'},
            {'name': 'body', 'type': 'CHARACTER'},
            {'name': 'homedest', 'type': 'CHARACTER'}
        ]
    }


def mock_get_operations(*args):
    return [
        {"id": 2100, "slug": "read-data", "ports": []},
        {"id": 2101, "slug": "cast", "ports": []},
        {"id": 2102, "slug": "rename", "ports": []},
        {"id": 2103, "slug": "discard", "ports": []},
        {"id": 2104, "slug": "find-replace", "ports": []},
        {"id": 2105, "slug": "sort", "ports": []},
        {"id": 2106, "slug": "filter", "ports": []},
        {"id": 2107, "slug": "group", "ports": []},
        {"id": 2108, "slug": "join", "ports": []},
        {"id": 2109, "slug": "concat-rows", "ports": []},
        {"id": 2110, "slug": "sample", "ports": []},
        {"id": 2111, "slug": "limit", "ports": []},
        {"id": 2112, "slug": "window-function", "ports": []},
        {"id": 2113, "slug": "python-code", "ports": []},
        {"id": 2114, "slug": "add-by-formula", "ports": []},
        {"id": 2115, "slug": "invert-boolean", "ports": []},
        {"id": 2116, "slug": "rescale", "ports": []},
        {"id": 2117, "slug": "round-number", "ports": []},
        {"id": 2118, "slug": "bucketize", "ports": []},
        {"id": 2119, "slug": "normalize", "ports": []},
        {"id": 2120, "slug": "force-range", "ports": []},
        {"id": 2121, "slug": "ts-to-date", "ports": []},
        {"id": 2122, "slug": "to-upper", "ports": []},
        {"id": 2123, "slug": "to-lower", "ports": []},
        {"id": 2124, "slug": "capitalize", "ports": []},
        {"id": 2125, "slug": "remove-accents", "ports": []},
        {"id": 2126, "slug": "normalize-text", "ports": []},
        {"id": 2127, "slug": "concat-attribute", "ports": []},
        {"id": 2128, "slug": "trim", "ports": []},
        {"id": 2129, "slug": "truncate-text", "ports": []},
        {"id": 2130, "slug": "split-into-words", "ports": []},
        {"id": 2131, "slug": "substring", "ports": []},
        {"id": 2132, "slug": "parse-to-date", "ports": []},
        {"id": 2133, "slug": "extract-numbers", "ports": []},
        {"id": 2134, "slug": "extract-with-regex", "ports": []},
        {"id": 2135, "slug": "extract-from-array", "ports": []},
        {"id": 2136, "slug": "concat-array", "ports": []},
        {"id": 2137, "slug": "create-array", "ports": []},
        {"id": 2138, "slug": "change-array-type", "ports": []},
        {"id": 2139, "slug": "sort-array", "ports": []},
        {"id": 2140, "slug": "force-date-range", "ports": []},
        {"id": 2141, "slug": "update-hour", "ports": []},
        {"id": 2142, "slug": "truncate-date-to", "ports": []},
        {"id": 2143, "slug": "date-diff", "ports": []},
        {"id": 2144, "slug": "date-add", "ports": []},
        {"id": 2145, "slug": "date-part", "ports": []},
        {"id": 2146, "slug": "format-date", "ports": []},
        {"id": 2147, "slug": "date-to-ts", "ports": []},
        {"id": 2148, "slug": "escape-xml", "ports": []},
        {"id": 2149, "slug": "escape-unicode", "ports": []},
        {"id": 2150, "slug": "stemming", "ports": []},
        {"id": 2151, "slug": "n-grams", "ports": []},
        {"id": 2152, "slug": "string-indexer", "ports": []},
        {"id": 2153, "slug": "one-hot-encoding", "ports": []},
        {"id": 2154, "slug": "flag-in-range", "ports": []},
        {"id": 2155, "slug": "flag-invalid", "ports": []},
        {"id": 2156, "slug": "flag-empty", "ports": []},
        {"id": 2157, "slug": "flag-with-formula", "ports": []},
        {"id": 2158, "slug": "clean-missing", "ports": []},
        {"id": 2159, "slug": "remove-missing", "ports": []},
        {"id": 2160, "slug": "handle-invalid", "ports": []},
        {"id": 2161, "slug": "remove-invalid", "ports": []},
        {"id": 2162, "slug": "duplicate", "ports": []},
        {"id": 2163, "slug": "select", "ports": []},
        {"id": 2164, "slug": "save", "ports": []},
        {"id": 2165, "slug": "replace-with-regex", "ports": []},
    ]
