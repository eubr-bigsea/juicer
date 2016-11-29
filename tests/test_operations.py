import pytest
import context
from juicer.spark.etl_operation import RandomSplit, Sort, Distinct, Sample, \
    Intersection, Difference
from juicer.spark.data_operation import DataReader
from juicer.spark.operation import Union
from juicer.spark.operation import Sort
from juicer.spark.operation import Distinct
from juicer.spark.operation import Sample
from juicer.spark.operation import Intersection
from juicer.spark.operation import Difference

#class Test:
#    def __init__(self):
#        self.parameters = {}
#        self.class_name = None
#
#    def test_success_minimal_parameters(self):
#        self.class_name(self, self.parameters,'in','out')


 
class TestDataReaderOperation:
    def test_success_minimal_parameters(self):
        self.parameters = {
            'infile':'file',
            'has_header':'True',
            'sep':','
        }
        self.inputs = []
        self.outputs = ['output_1']
        self.class_name = DataReader
        instance = self.class_name(self.parameters, self.inputs, self.outputs)
        assert instance.generate_code() == \
            "output_1 = spark.read.csv('file', header=True, sep=',')"



class TestRandomSplit:
    def test_success_minimal_parameters(self):
        self.parameters = {
            'weights': [2,3],
            'seed':'0'
        }
        self.inputs = ['input_1']
        self.outputs = ['output_1','output_2']
        self.class_name = RandomSplit
        instance =  self.class_name(self.parameters, self.inputs, self.outputs)
        assert instance.generate_code() == \
            "output_1, output_2 = input_1.randomSplit([2.0, 3.0], 0)"



class TestUnion:
    def test_success_minimal_parameters(self):
        self.parameters = {}
        self.inputs = ['input_1', 'input_2']
        self.outputs = ['output_1']
        self.class_name = Union
        instance =  self.class_name(self.parameters, self.inputs, self.outputs)
        assert instance.generate_code() == \
            "output_1 = input_1.unionAll(input_2)"



class TestSortOperation:
    def test_success_minimal_parameters(self):
        self.parameters = {
            'columns': ["name", "class"],
            'ascending': ["1", "0"]
        }
        self.inputs = ['input_1']
        self.outputs = ['output_1']
        self.class_name = Sort
        instance = self.class_name(self.parameters, self.inputs, self.outputs)
        assert instance.generate_code() == 'output_1 = input_1.orderBy([{}], ascending=[{}])'.format(
            ', '.join('"{}"'.format(x) for x in self.parameters['columns']), 
            ', '.join(x for x in self.parameters['ascending']))

#    def test_error_missing_parameters(self):
#        sort = Sort({}, 'in_df', 'out_df')
#        with pytest.raises(KeyError, message="Expecting columns value in parameters"):
#            sort.generate_code()
#
#    def test_error_incorrect_df_names(self):
#        sort = Sort({}, 'in_df', 'out_df')
#        assert sort.generate_code() == ''


class TestDistinct:
    def test_success_minimal_parameters(self):
        self.parameters = {}
        self.inputs = ['input_1']
        self.outputs = ['output_1']
        self.class_name = Distinct
        instance =  self.class_name(self.parameters, self.inputs, self.outputs)
        assert instance.generate_code() == \
            "output_1 = input_1.distinct()"


class TestSample:
    def test_success_minimal_parameters(self):
        self.parameters = {
            'withReplacement':'False',
            'fraction':'0.3',
            'seed':'0'
        }
        self.inputs = ['input_1']
        self.outputs = ['output_1']
        self.class_name = Sample
        instance =  self.class_name(self.parameters, self.inputs, self.outputs)
        assert instance.generate_code() == \
            "output_1 = input_1.sample(withReplacement={}, fraction={}, seed={})".format(
                self.parameters['withReplacement'], self.parameters['fraction'],
                self.parameters['seed'])


class TestIntersection:
    def test_success_minimal_parameters(self):
        self.parameters = {}
        self.inputs = ['input_1','input_2']
        self.outputs = ['output_1']
        self.class_name = Intersection
        instance =  self.class_name(self.parameters, self.inputs, self.outputs)
        assert instance.generate_code() == \
            "output_1 = input_1.intersect(input_2)"


class TestDifference:
    def test_success_minimal_parameters(self):
        self.parameters = {}
        self.inputs = ['input_1','input_2']
        self.outputs = ['output_1']
        self.class_name = Difference
        instance =  self.class_name(self.parameters, self.inputs, self.outputs)
        assert instance.generate_code() == \
            "output_1 = input_1.subtract(input_2)"
