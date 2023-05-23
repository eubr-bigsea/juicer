import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error

from juicer.explainable_ai.understand_ai import Understanding


class Interpretation(Understanding):

    def __init__(self,
                 arguments_used,
                 model_to_understand=None,
                 data_source=None,
                 feature_names=None,
                 target_name=None,
                 feature_importance=None):

        """

        :param arguments_used:
        :param model_to_understand:
        :param data_source:
        :param feature_names:
        :param target_name:
        :param feature_importance:
        """

        super().__init__(arguments_used, model_to_understand, data_source, feature_names, target_name)
        self.feature_importance = feature_importance

    @property
    def feature_importance(self):
        return self._feature_importance

    @feature_importance.setter
    def feature_importance(self, feature_importance):
        if feature_importance is not None:
            self._feature_importance = feature_importance
        elif self.model_to_understand is not None:
            if hasattr(self.model_to_understand, 'feature_importances_'):
                self._feature_importance = self.model_to_understand.feature_importances_
            elif hasattr(self.model_to_understand, 'coef_'):
                self._feature_importance = self.model_to_understand.coef_
            else:
                raise ValueError(f'Model {self.model_to_understand.__class__.__name__} does not have'
                                 f'interpretable attribute in {self.__class__.__name__}')
        else:
            raise ValueError(f'{self.__class__.__name__} class does not find feature importance.')

    def _uai_feature_importance(self, *args, **kwargs):
        n_feature = kwargs.get('n_feature')
        if n_feature:
            sorted_idx = np.argsort(self.feature_importance)[::-1]
            n_imp = [self.feature_importance[i] for i in sorted_idx[:n_feature]]
            n_names = [self.feature_names[i] for i in sorted_idx[:n_feature]]
            return n_imp, n_names
        else:
            return self.feature_importance, self.feature_names


class TreeInterpretation(Interpretation):

    def __init__(self,
                 arguments_used,
                 model_to_understand=None,
                 data_source=None,
                 feature_names=None,
                 target_name=None,
                 feature_importance=None):
        super().__init__(arguments_used,
                         model_to_understand,
                         data_source,
                         feature_names,
                         target_name,
                         feature_importance)

    def _uai_dt_surface(self, *args, **kwargs):
        max_deep = kwargs.get('max_deep')
        return {'feature_names': self.feature_names,
                'max_deep': max_deep,
                'model': self.model_to_understand}


class EnsembleInterpretation(Interpretation):

    def __init__(self,
                 arguments_used,
                 model_to_understand=None,
                 data_source=None,
                 feature_names=None,
                 target_name=None,
                 feature_importance=None):
        super().__init__(arguments_used,
                         model_to_understand,
                         data_source,
                         feature_names,
                         target_name,
                         feature_importance)

    def _uai_forest_importance(self, *args, **kwargs):
        n_feature = kwargs.get('n_feature')
        std = np.std([tree.feature_importances_ for tree in self.model_to_understand.estimators_], axis=0)
        if n_feature:
            sorted_idx = np.argsort(self.feature_importance)[::-1]
            n_std = [std[i] for i in sorted_idx[:n_feature]]
            n_imp = [self.feature_importance[i] for i in sorted_idx[:n_feature]]
            n_names = [self.feature_names[i] for i in sorted_idx[:n_feature]]
            return n_imp, n_std, n_names
        else:
            return self.feature_importance, std, self.feature_names


class LinearRegressionInterpretation(Interpretation):

    def __init__(self,
                 arguments_used,
                 model_to_understand=None,
                 data_source=None,
                 feature_names=None,
                 target_name=None,
                 feature_importance=None):

        super().__init__(arguments_used,
                         model_to_understand,
                         data_source,
                         feature_names,
                         target_name,
                         feature_importance)

    def _uai_p_value(self, *args, **kwargs):
        if self.data_source is not None:
            X = self.data_source[self.feature_names[:-1]].values
            if self.target_name:
                y = self.data_source[self.target_name].values
            else:
                y = self.data_source.iloc[:, -1:].values.reshape(1, -1)[0]
        else:
            raise ValueError(f'{self.__class__.__name__} class must have a data source in order to calculate p-value.')

        if self.model_to_understand:
            coef = np.append(self.model_to_understand.intercept_, self.model_to_understand.coef_)
            X = np.append(np.ones((len(X), 1)), X, axis=1)
        else:
            raise ValueError(f'{self.__class__.__name__} class must have a model')

        predictions = self.model_to_understand.predict(self.data_source[self.feature_names[:-1]])
        mse = mean_squared_error(y, predictions)
        var_b = mse*(np.linalg.pinv(np.dot(X.T, X)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = coef/sd_b
        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(X) - len(X[0])))) for i in ts_b]

        return pd.DataFrame({
            "Coefficients": np.round(coef, 4),
            "Standard Errors": np.round(sd_b, 3),
            't_values': np.round(ts_b, 3),
            'p_values': np.round(p_values, 3)
        })

















