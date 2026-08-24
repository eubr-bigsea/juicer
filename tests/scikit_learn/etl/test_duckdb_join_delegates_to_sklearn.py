import juicer.scikit_learn.duckdb.etl_operation as duckdb_etl
import juicer.scikit_learn.etl_operation as sk_etl


# Demo 2: DuckDB's JoinOperation used to be a byte-for-byte copy of the
# scikit-learn base class's generate_code(). It is now a plain alias
# (`JoinOperation = sk.JoinOperation` in
# juicer/scikit_learn/duckdb/etl_operation.py). These tests prove the
# alias is a real identity (not just an equivalent reimplementation) and
# that instances built through either import path behave identically.
def test_duckdb_join_is_the_same_class_as_sklearn_join():
    assert duckdb_etl.JoinOperation is sk_etl.JoinOperation


def test_duckdb_join_generates_same_code_as_sklearn_join():
    arguments = {
        'parameters': {'keep_right_keys': False, 'match_case': True,
                       'join_type': 'inner',
                       'left_attributes': ['name'],
                       'right_attributes': ['name']},
        'named_inputs': {
            'input data 1': 'df1',
            'input data 2': 'df2'
        },
        'named_outputs': {
            'output data': 'out'
        }
    }
    sk_instance = sk_etl.JoinOperation(**arguments)
    duckdb_instance = duckdb_etl.JoinOperation(**arguments)

    assert duckdb_instance.generate_code() == sk_instance.generate_code()
