def parallel_fit_tasks(est, train, eva, validation, epm, collect_sub_model):
    """
    Copied from
    https://fossies.org/linux/misc/spark-2.4.0.tgz/spark-2.4.0/python/pyspark/ml/tuning.py
    """
    model_iter = est.fitMultiple(train, epm)

    def single_task():
        index, model = next(model_iter)
        metric = eva.evaluate(model.transform(validation, epm[index]))
        return index, metric, model if collect_sub_model else None

    return [single_task] * len(epm)


def cross_validation(df, fold_col, estimator, estimator_params,
                            evaluator, collect_sub_models,
                            processes):
    """
    Performs cross-validation in Lemonade
    """
    from multiprocessing.pool import ThreadPool
    from pyspark.ml.tuning import CrossValidatorModel
    import numpy as np
    from pyspark.sql import functions as f

    num_models = len(estimator_params)
    avg_metrics_by_fold = [0.0] * num_models
    folds = 1 + df.agg(f.max(fold_col)).head()[0]  # fold_col is zero based

    pool = ThreadPool(processes=min(processes, num_models))
    sub_models = None
    if collect_sub_models:
        sub_models = [[None for j in range(num_models)] for i in range(folds)]

    metrics = [[] for i in range(folds)]
    for i in list(range(folds)):
        condition = (df[fold_col] == i)
        validation = df.filter(condition).cache()
        train = df.filter(~condition).cache()
        tasks = parallel_fit_tasks(estimator, train, evaluator, validation,
                                   estimator_params,
                                   collect_sub_models)
        for j, metric, subModel in pool.imap_unordered(lambda op: op(), tasks):
            avg_metrics_by_fold[j] += (metric / folds)
            metrics[i].append(metric)
            if collect_sub_models:
                sub_models[i][j] = subModel

        validation.unpersist()
        train.unpersist()

    if evaluator.isLargerBetter():
        best_index = np.argmax(avg_metrics_by_fold)
    else:
        best_index = np.argmin(avg_metrics_by_fold)
    best_model = estimator.fit(df, estimator_params[best_index])
    # Return tuple with Model and avg_metrics_by_fold
    return CrossValidatorModel(best_model, avg_metrics_by_fold,
                               sub_models), metrics, best_index, folds
