import numpy as np
from scipy.stats import qmc
import diversipy
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import pandas as pd


def scale(points, scaler):
    """Return points in scaled units"""
    return scaler.transform(points)


def unscale(points, scaler):
    """Return points in original units"""
    return scaler.inverse_transform(points)


def sample_space(data_length, data_points, bounds=None):
    """Sample parameter space, will return results between bounds (default between 0-1)"""

    sampler = qmc.Sobol(d=data_points, scramble=False)
    sample = sampler.random_base2(m=int(np.log2(data_length)))

    # Scale if desired
    if np.any(bounds):
        sample = qmc.scale(sample, bounds[0, :], bounds[1, :])

    return sample


def diversipy_sampling(x_cloud, n_sample, fraction_greedy=0.25, fraction_psa=0.75):
    """Uniformly sample n_sample points from a 2D point cloud, samples rows.
    Sample n_sim points from the (refilled) NROY region using a mix of greedy_minmax and psa_select
    methods from diversipy to balance filling the space against obtaining enough points around the edges"""

    return np.concatenate((
        diversipy.subset.select_greedy_maximin(x_cloud, int(fraction_greedy*n_sample)),
        diversipy.subset.psa_select(x_cloud, int(fraction_psa*n_sample))
    ), axis=0)


def sort_nicely(l_in):
    """ Sort the given list in the way that humans expect.
    """
    l_in.sort(key=alphanum_key)
    return l_in


def label(x, color, label, textcolor=None, fontweight="bold", fontsize=None):
    """Define and use a simple function to label the plot in axes coordinates"""
    if textcolor is not None:
        color = textcolor
    ax = plt.gca()
    ax.text(0, .2, label, fontweight=fontweight, color=color, fontsize=fontsize,
            ha="left", va="center", transform=ax.transAxes)


def try_int(s_in):
    try:
        return int(s_in)
    except:
        return s_in


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [try_int(c) for c in re.split('([0-9]+)', s)]


def get_cat_colors(y_cats):
    """Create dictionary where keys are readout categories"""
    y_cats_unique = tuple(Counter(y_cats).keys())
    colors = sns.color_palette("cubehelix", len(y_cats_unique), as_cmap=False)
    return dict(zip(y_cats_unique, colors))


def mahalanobis_outliers(data, percentile=0.95):
    """Identify outliers based on Mahalanobis distance"""

    # Remove data columns with all zeros from consideration
    data = data[:, ~np.all(data == 0, axis=0)]

    outliers = []
    x_minus_mu = data - np.nanmean(data, axis=0)
    cov = np.cov(data.T)  # Covariance
    inv_covmat = sp.linalg.inv(cov)  # Inverse covariance
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())

    c = np.sqrt(sp.stats.chi2.ppf(percentile, df=data.shape[1]))  # degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > c:
            outliers.append(index)

    return outliers


def mad_outliers(data, m_outlier=3.0):
    """Identify outliers based on median absolute deviation"""

    # Remove data columns with all zeros from consideration
    data = data[:, ~np.all(data == 0, axis=0)]

    outliers = []

    for i in range(data.shape[1]):
        # Compute mean absolute deviation (MAD) for each output
        mad = np.nanmedian(abs(data[:, i] - np.nanmedian(data[:, i])))

        # Find outliers
        med = np.abs(data[:, i] - np.nanmedian(data[:, i])) / mad
        if np.any(med > m_outlier):
            outliers.extend(list(np.where(med > m_outlier)[0]))

    # Keep only unique outliers and store the indices of not-outliers
    return list(set(outliers))


def create_log(log_file, x_names, x_limits, y_names, y_observed, sigma_observed, constants):
    """Create log file, summary of wave data and parameters to txt file"""

    # Loop through each row of self.x_limits and print parameter range to string
    x_pars_str = "-----------\nParameters:\n-----------\n"
    for i in range(len(x_names)):
        x_pars_str += "- " + str(x_names[i]) + ": [" + str(x_limits[0, i]) + " - " + \
                        str(x_limits[1, i]) + "]\n"

    # Loop through each row of self.y_limits and print parameter range to string
    y_data_str = "\n-------------\nObservations:\n-------------\n"
    for i in range(len(y_names)):
        y_data_str += "- " + str(y_names[i]) + ": " + str(y_observed[i]) +\
                      " ± " + str(sigma_observed[i]) + "\n"

    constants_str = "\n-------------\nConstants:\n-------------\n"
    if constants is None:
        constants_str += "None specified\n"
    else:
        for key, value in constants.items():
            constants_str += "- " + str(key) + ": " + str(value) + "\n"

    print(x_pars_str)
    print(y_data_str)
    print(constants_str)

    # Export self.x_names to txt file and replace file if existing
    with open(log_file, "w") as f:
        f.write(x_pars_str)
        f.write(y_data_str)
        f.write(constants_str)


def update_log(log_file, str_log):
    """Print and add str to log file"""
    print(str_log)
    with open(str(log_file), "a") as f:
        f.write(str_log + "\n")


def normalize_data_ci(y, mean, std, n_std=2.0):
    """Normalize y based on mean and n_std*std of data, translate mean to 0.0 and confidence intervals at +-0.5"""
    return (y - (mean - n_std*std)) / ((mean + n_std*std) - (mean - n_std*std)) - 0.5


def convert_data(df_mean, df_std, meta=None, growth=False, csv_dir=None, csv_name=None):
    """Convert wide format mean and standard deviation data to format used in the Wave class"""

    # Convert to pandas if not already
    if not isinstance(df_mean, pd.DataFrame):
        df_mean = pd.DataFrame(df_mean)
    if not isinstance(df_std, pd.DataFrame):
        df_std = pd.DataFrame(df_std)

    if growth:
        # Add day labels to column names
        df_mean.index = ["mu_d" + str(i) for i in df_mean.index]
        df_std.index = ["sigma_d" + str(i) for i in df_std.index]

        # Combine mean and standard deviation in one dataframe
        data = pd.concat([df_mean, df_std]).T

    else:
        # Combine mean, std, and limits in one dataframe and switch columns and rows
        data = pd.concat([df_mean, df_std]).T

        # Change column names to mu and sigma and add min and max if limits are given
        if df_mean.shape[0] == 1:
            data.columns = ["mu", "sigma"]
        elif df_mean.shape[0] == 2:
            data.columns = ["mu_baseline", "mu_acute", "sigma_baseline", "sigma_acute"]
        else:
            raise ValueError("mean must have 1 or 2 rows")

    # If any metadata is provided
    if meta is not None:
        # Add physiological limits
        if "min" and "max" in meta.index:
            data = pd.concat([data, meta.loc[["min", "max"]].T], axis=1)
            data.columns = list(data.columns[:-2]) + ["min", "max"]
        # Add data categories
        if "category" in meta.index:
            data = pd.concat([data, meta.loc["category"].T], axis=1)
            data.columns = list(data.columns[:-1]) + ["category"]

    # Export to csv, create directory if nonexistent, if directory is not given export to current directory
    if csv_name is not None:
        if csv_dir is None:
            csv_dir = ""
        else:
            csv_dir.mkdir(parents=True, exist_ok=True)
        # If filename does not end with .csv, add it
        if not csv_name.endswith(".csv"):
            csv_name += ".csv"
        data.to_csv(csv_dir / csv_name)

    return data
