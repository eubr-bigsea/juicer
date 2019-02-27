# coding=utf-8
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer, VectorAssembler, StringIndexer, \
    IndexToString
from pyspark.ml.linalg import VectorUDT

from juicer.util import dataframe_util


def assemble_features_pipeline_model(df, features, label, algorithm,
                                     set_features, set_label, prediction, keep,
                                     emit, task_id):
    """
    Prepare features and label to be processed by a ML algorithm. Features and
    labels are indexed (StringIndexer) if they are categorical.
    During the process, temporary columns are created but they're removed as
    soon as the pipeline ends.
    :arg df Input data frame
    :arg features array with column names to be used as features
    :arg label name of the column with label
    :arg algorithm algorithm to be used, can be a ML or a feature extraction one
    :arg set_features name of the method used to set the features
    :arg set_label name of the method used to set the label
    :arg prediction name of the prediction column (generated)
    :arg keep list of the columns to be kept after the processing
    :arg emit emit messages function
    :arg task_id task identifier
    :returns processing pipeline model
    """
    if keep is None:
        keep = []
    final_keep = [c.name for c in df.schema]
    final_keep.extend(keep)

    clean_null_rows = 'SELECT * FROM __THIS__ WHERE {}'
    if len(features) > 1 and not isinstance(
            df.schema[str(features[0])].dataType, VectorUDT):

        emit(name='update task',
             message=_('Features are not assembled as a vector. They will be '
                       'implicitly assembled and rows with null values will be '
                       'discarded. If this is undesirable, explicitly add a '
                       'attribute vectorizer, handle missing data and '
                       'categorical attributes in the workflow.'),
             level='warning', status='RUNNING', identifier=task_id)
        stages = []
        to_assemble = []
        for f in features:
            if not dataframe_util.is_numeric(df.schema, f):
                name = f + '__tmp__'
                to_assemble.append(name)
                stages.append(StringIndexer(
                    inputCol=f, outputCol=name, handleInvalid='keep'))
            else:
                to_assemble.append(f)

        # Remove rows with null (VectorAssembler doesn't support it)
        cond = ' AND '.join(['{} IS NOT NULL '.format(c)
                             for c in to_assemble])
        stages.append(SQLTransformer(statement=clean_null_rows.format(cond)))

        final_features = 'features__tmp__'
        stages.append(VectorAssembler(
            inputCols=to_assemble, outputCol=final_features))

        getattr(algorithm, set_features)(final_features)

        if label is not None:
            final_label = '{}__tmp__'.format(label)
            getattr(algorithm, set_label)(final_label)
            stages.append(StringIndexer(inputCol=label, outputCol=final_label,
                                        handleInvalid='keep'))

        stages.append(algorithm)
        pipeline = Pipeline(stages=stages)
        model = pipeline.fit(df)

        last_stages = [model]
        if label is not None:
            last_stages.append(IndexToString(inputCol=prediction,
                                             outputCol='{}'.format(prediction),
                                             labels=model.stages[-2].labels))

        # Remove temporary columns
        sql = 'SELECT {} FROM __THIS__'.format(', '.join(final_keep))
        last_stages.append(SQLTransformer(statement=sql))

        pipeline = Pipeline(stages=last_stages)
        model = pipeline.fit(df)

    else:
        if label is not None:
            final_label = '{}__tmp__'.format(label)

            getattr(algorithm, set_label)(final_label)
            stages = [
                StringIndexer(inputCol=label, outputCol=final_label,
                              handleInvalid='keep'),
                algorithm
            ]

            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(df)
            last_stages = [model]
            if label is not None:
                last_stages.append(IndexToString(inputCol=final_label,
                                                 outputCol='{}_str'.format(
                                                     prediction),
                                                 labels=model.stages[
                                                     -2].labels))

            # Remove temporary columns
            sql = 'SELECT {} FROM __THIS__'.format(', '.join(final_keep))
            last_stages.append(SQLTransformer(statement=sql))

            pipeline = Pipeline(stages=last_stages)
            model = pipeline.fit(df)

        else:
            getattr(algorithm, set_features)(features[0])
            model = algorithm.fit(df)

    return model
