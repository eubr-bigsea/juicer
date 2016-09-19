import pytest
from operation import Sort


class TestSortOperation:

    def test_error_missing_parameters(self):
        sort = Sort({}, 'in_df', 'out_df')
        with pytest.raises(KeyError, message="Expecting columns value in parameters"):
            sort.generate_code()

    def test_success_minimal_parameters(self):
        params = {
            'columns': ["name", "class"],
            'ascending': [True, False]
        }
        sort = Sort(params, ['in_df'], ['out_df'])
        assert sort.generate_code() == 'out_df = in_df.orderBy([{}], ascending = [{}])'.format(
            ', '.join('"{}"'.format(x) for x in params['columns']), 
            ', '.join(str(x).lower() for x in params['ascending']))

    def test_error_incorrect_df_names(self):
        sort = Sort({}, 'in_df', 'out_df')
        assert sort.generate_code() == ''
        
