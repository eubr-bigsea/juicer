# coding=utf-8

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def coxph(data, y_name, time_name, x_names):
    x = {y_name: data[y_name] == 1, time_name: data[time_name]}

    for x_name in x_names:
        for factor in set(data[x_name]):
            if factor == min(set(data[x_name])):
                continue
            x[x_name + str(factor)] = data[x_name] == factor
    x = pd.DataFrame(x)
    cph = CoxPHFitter()
    return x, cph.fit(x, duration_col=time_name, event_col=y_name)


# noinspection PyUnresolvedReferences
def get_schoenfeld_residuals(y, x, event, params, covar):
    """
    This is the function that returns scaled schoenfeld residuals.

    :param y: the 'time' variable
    :param x: is the n by p data matrix
    :param event:
    :param params: is the parameter values from the Cox proportional hazard
        model
    :param covar: is the variance-covariance matrix from the Cox proportional
    :return:
    """

    # Create a dataframe to hold the scaled Schoenfeld residuals
    schoenfeld_residuals = pd.DataFrame(columns=[x.columns])

    # Create a dataframe to hold the 'Time variable'
    schoenfeld_time = pd.DataFrame(columns=['Time'])

    # Add the 'Time' variable to the data matrix 'X'. This will be
    # useful to select units still at risk of event
    x['Time'] = y

    # Add the 'event' variable to the data matrix 'X'. This will be
    # useful to select units who experienced the event
    x['Eventoccured'] = event

    # Sort 'X' based on time (ascending order)
    x = x.sort_values(['Time'])

    # Get the number of units
    number_of_units = len(x)
    # Set the counter to zero
    counter = 0

    # Get the number of units that experienced the event
    number_of_events = np.sum(event)

    # For each unit, calculate the residuals if they experienced the event
    for patient_index in range(number_of_units):
        if x['Eventoccured'].iloc[patient_index] == 1:
            current_time = x['Time'].iloc[patient_index]

            # Sum of the hazards for all the observations still at risk
            sum_hazards = np.sum(np.exp(np.dot(
                x.loc[x['Time'] >= current_time].iloc[:, :len(x.columns) - 2],
                params)))

            # Calculate the probability of event for each unit still at risk
            probability_of_death_all = np.ravel(np.exp(np.dot(
                x.loc[x['Time'] >= current_time].iloc[:, :len(x.columns) - 2],
                params)) / sum_hazards)
            # Calculate the expected covariate values
            expected_covariate_values = np.dot(probability_of_death_all, x.loc[(
                x['Time'] >= current_time)].iloc[:, :len(x.columns) - 2])

            # Get Schoenfeld residuals as the difference between
            # the current unit's covariate values and the expected covariate
            # values calculated from all units at risk
            residuals = (x.iloc[patient_index, :len(x.columns) - 2] -
                         expected_covariate_values)

            # Scale the residuals by the variance-covariance matrix of model
            # parameters
            scaled_residuals = number_of_events * np.dot(covar, residuals)

            # Add the scaled residuals to the dataframe for residuals
            schoenfeld_residuals.loc[counter] = scaled_residuals

            # Add the current time for the current unit. This can be used to
            # regress scaled residuals against time
            schoenfeld_time.loc[counter] = current_time
            counter += 1

    schoenfeld_residuals['Time'] = schoenfeld_time
    return schoenfeld_residuals


def cox(data, y_name, t_name, columns):
    x, cf = coxph(data, y_name, t_name, columns)

    # noinspection PyProtectedMember
    hessian = np.linalg.inv(cf._hessian_)

    x = x.drop(y_name, 1)
    # noinspection PyUnresolvedReferences
    x = x.drop(t_name, 1)

    return get_schoenfeld_residuals(data[t_name], x, data[y_name],
                                    cf.hazards_.transpose(), hessian[0])
