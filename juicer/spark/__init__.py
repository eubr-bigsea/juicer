# coding=utf-8


def spark_summary_translations(v):
    """ This is used to extract messages that are provided by Spark.
    """
    mapping = {
        'coefficientStandardErrors': _('Coefficient standard errors'),
        'degreesOfFreedom': _('Degrees of freedom'),
        'devianceResiduals': _('Deviance residuals'),
        'explainedVariance': _('Explained variance'),
        'featuresCol': _('Features attribute'),
        'labelCol': _('Label attribute'),
        'meanAbsoluteError': _('Mean absolute error'),
        'meanSquaredError': _('Mean squared error'),
        'numInstances': _('Number of instances'),
        'objectiveHistory': _('Objective history'),
        'pValues': _('P-values'),
        'predictionCol': _('Prediction attribute'),
        'predictions': _('Predictions'),
        'r2': _(u'R²'),
        'r2adj': _(u'Adjusted R²'),
        'residuals': _('Residuals'),
        'rootMeanSquaredError': _('Root mean squared error'),
        'tValues': _('T-values'),
        'totalIterations': _('Total of iterations'),
    }
    return mapping.get(v.strip(), '???')
