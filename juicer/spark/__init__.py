# coding=utf-8


def spark_summary_translations(v):
    """ This is used to extract messages that are provided by Spark.
    """
    mapping = {
        'aic': _('Akaike\'s "An Information Criterion" (AIC)'),
        'coefficientStandardErrors': _('Coefficient standard errors'),
        'degreesOfFreedom': _('Degrees of freedom'),
        'depth': _('Depth'),
        'devianceResiduals': _('Deviance residuals'),
        'deviance': _('Deviance'),
        'dispersion': _('Dispersion'),
        'explainedVariance': _('Explained variance'),
        'featuresCol': _('Features attribute'),
        'featureImportances': _('Feature importances'),
        'getNumTrees': _('Number of trees'),
        'labelCol': _('Label attribute'),
        'meanAbsoluteError': _('Mean absolute error'),
        'meanSquaredError': _('Mean squared error'),
        'nullDeviance': _('Deviance for the null model'),
        'numClasses': _('Number of classes'),
        'numFeatures': _('Number of features'),
        'numNodes': _('Number of nodes'),
        'numInstances': _('Number of instances'),
        'numIterations': _('Iterations'),
        'objectiveHistory': _('Objective history'),
        'pValues': _('P-values'),
        'predictionCol': _('Prediction attribute'),
        'predictions': _('Predictions'),
        'r2': _('R²'),
        'r2adj': _('Adjusted R²'),
        'rank': _('Rank'),
        'residuals': _('Residuals'),
        'residualDegreeOfFreedom': _('Residual degree of freedom'),
        'residualDegreeOfFreedomNull': _('Residual degree of freedom for the null model'),
        'rootMeanSquaredError': _('Root mean squared error'),
        'solver': _('Solver'),
        'trees': _('Trees'),
        'tValues': _('T-values'),
        'totalIterations': _('Total of iterations'),
    }
    return mapping.get(v.strip(), v)
