from inspect import getmembers

import pandas as pd


class Understanding:

    """

    """

    def __init__(self, arguments_used: dict,
                 model_to_understand,
                 data_source: pd.DataFrame,
                 feature_names=None,
                 target_name=None):
        """

        :param arguments_used: A dictionary containing all informative arguments used in this complex model understanding.
        :param model_to_understand: A scikit-learn trained model.
        :param data_source: A pandas DataFrame.
        :param feature_names: An iterable containing feature names. If feature_names is None, the column names in
        pandas DataFrame will be used.
        :param target_name: A string indicating the name of the target column in the data source.
        If target_name is None, the last column of the data source will be considered as the target.
        :param generated_args_dict: A dictionary with all the informative resource generated by XAI strategy.
        """

        self.uai_members = self._get_uai_members()
        self.arguments_used = arguments_used
        self.model_to_understand = model_to_understand
        self.data_source = data_source
        self.target_name = target_name
        self.feature_names = feature_names
        self.generated_args_dict = None



    @property
    def arguments_used(self):
        return self._arguments_used

    @arguments_used.setter
    def arguments_used(self, arguments_used):
        for arg in arguments_used:
            if arg not in self.uai_members:
                # fail fast (python concept)
                raise ValueError(f"UAI argument: {arg} don't exist in {self.__class__.__name__}")
        self._arguments_used = arguments_used

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names):
        if feature_names:
            self._feature_names = feature_names
        elif isinstance(self.data_source, pd.DataFrame):
            if self.target_name is None:
                self._feature_names = self.data_source.keys().tolist()
            else:
                raise NotImplementedError('TODO select datasource target')
        elif isinstance(self.data_source, pd.Series):
            self._feature_names = [self.data_source.name]
        else:
            raise ValueError(f'{self.__class__.__name__} class does not have features name')

    def _get_uai_members(self):
        members = getmembers(self)
        return {name.strip('_uai_'): member for name, member in members if name.startswith('_uai')}

    def generate_arguments(self):
        """
        This method execute all understander routines and create a dictionary with results.

        """
        self.generated_args_dict = {arg: self.uai_members[arg](**param) for arg, param in self.arguments_used.items()}

    def _uai_feature_importance(self):
        pass



