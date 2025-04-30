import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, metrics
import random
import time
import pickle
from SALib.analyze import sobol
from SALib.sample import saltelli
import torch
import gpytorch
import glob
import shutil
from datetime import datetime
import warnings
from .gpe_pytorch import ExactGPModel, MultitaskGPModel
from .utils import scale, unscale, sample_space, sort_nicely, diversipy_sampling, create_log, update_log, clean_up


class Camouflage:
    """
    Single wave of Bayesian history matching process. This class is suitable to a multivariate input and output problem.
    It was originally designed for use with cardiogrowth but can be used with any type of model. Mind that this method
    is not involved in the evaluation of the model itself, but rather gives inputs for the model and uses the model
    outputs to train GPEs that are used to determine the NROY region of the provided parameter space.
    """

    def __init__(self, wave_number, root_dir, threshold, pars, data_df, nroy=None, clear_log=False,
                 n_emu0=2 ** 20, n_emu_min=1e5, n_sim=2 ** 7, n_posterior=2 ** 8, plot_cols=4, log_file="bhm.log",
                 constants=None, prior_dir=None, prior_names=None, posterior_label="Posterior", clear_dir=False,
                 validation=False, x_posterior=None, y_posterior=None, os_compatibility=False,
                 sim_dir_name="sims", gpe_dir_name="gpe", print_log=True, save_log=True, wave_name=None):
        """ Initialize Wave """

        # Pass in to the class
        self.wave_number = wave_number
        self.name = "Wave " + str(wave_number + 1) if wave_name is None else wave_name
        self.threshold = threshold
        self.n_emu0 = int(n_emu0)
        self.n_emu_min = int(n_emu_min)
        self.n_sim = int(n_sim)
        self.n_posterior = int(n_posterior)
        self.nroy = nroy
        self.sobol_problem = None
        self.sobol_indices = None
        self.sim_time = 0
        self.constants = constants
        self.posterior_label = posterior_label
        self.os_compatibility = os_compatibility

        # Import data
        self.data_df = data_df
        self.y_names, self.y_observed, self.sigma_observed, self.y_lims, self.y_cats = self.import_data()

        # Calculate and store data dimensions
        self.n_x = len(pars)
        self.n_y = self.y_observed.size

        # Get parameter information
        self.x_names = []
        self.x_cats = []
        self.x_limits = np.empty((2, self.n_x))
        self.x_target = []
        for i_par, key in enumerate(pars):
            self.x_names.append(key)
            self.x_limits[:, i_par] = pars[key]["limits"]
            if "category" in pars[key]:
                self.x_cats.append(pars[key]["category"])
            if "target" in pars[key]:
                self.x_target.append(pars[key]["target"])

        # Disable categories and target if not provided
        if len(self.x_cats) == 0:
            self.x_cats = [str(el) for el in list(range(self.n_y))]
        if len(self.x_target) == 0:
            self.x_target = None

        # Create directory for wave and simulations
        if clear_dir:
            if root_dir.exists():
                shutil.rmtree(root_dir)
        self.dir = root_dir / self.name
        self.dir_sim = self.dir / sim_dir_name
        self.dir_gpe = self.dir / gpe_dir_name
        self.gpe_models = []
        self.gpe_likelihoods = []
        self.dir_sim.mkdir(exist_ok=True, parents=True)
        self.dir_gpe.mkdir(exist_ok=True, parents=True)

        # Check if log file does not exist in the root directory, if not create it
        self.log_file = root_dir / log_file
        self.print_log = print_log
        self.save_log = save_log
        if clear_log and save_log:
            create_log(self.log_file, self.x_names, self.x_limits, self.y_names, self.y_observed, self.sigma_observed,
                       self.constants, self.x_target, self.print_log)

        # Start wave log
        if save_log:
            update_log(self.log_file, "\n" + len(self.name) * "-" + "\n" + self.name + "\n" + len(self.name) * "-",
                       self.print_log)

        # If no NROY region is given (typically for first wave), fill up both spaces using Sobol sampling
        if self.nroy is None:
            if print_log:
                update_log(self.log_file, "Generated emulation point cloud with " + str(self.n_emu0) + " points",
                           self.print_log)
            self.x_emu = sample_space(n_emu0, self.n_x, self.x_limits)
            self.n_emu_init = self.x_emu.shape[0]
        # ... otherwise use NROY region from the previous wave
        else:
            self.x_emu = self.nroy
            self.n_emu_init = self.x_emu.shape[0]

        # If a prior pickle is provided, add prior samples to the emulation space, only do this once
        if prior_dir is not None:
            self.add_prior(prior_dir, prior_names)
        else:
            self.prior_names = []

        # Generate extra emulation points if number of emulation points is too low
        if self.n_emu_init < self.n_emu_min:
            if print_log:
                update_log(self.log_file, "Regenerating emulation cloud from " + str(self.x_emu.shape[0]) +
                           " to " + str(self.n_emu_min) + " points...", self.print_log)
            self.regenerate_emu()

        # Sample simulation points from the NROY region using diversipy to ensure uniform sampling
        if not validation:
            self.x_sim = diversipy_sampling(self.x_emu, self.n_sim)
        else:
            self.x_sim = []
        self.y_sim = np.empty(0)

        # Number of emulation points
        self.n_emu = self.x_emu.shape[0]

        # Split data sets and pre-allocate prediction y array
        self.x_training = self.y_training = self.x_validation = self.y_validation = \
            self.y_prediction = self.y_prediction_variance = np.empty(0)

        # Assign posterior samples if provided
        self.x_posterior = x_posterior
        self.y_posterior = y_posterior

        # Pre-allocate empty lists and arrays, reset the NROY region
        self.r_2 = np.zeros(self.n_y)
        self.implausibility = self.i_nroy = np.zeros(self.n_emu)
        self.y_emu = self.y_emu_variance = self.implausibility_y = np.empty(0)
        self.nroy_y = self.nroy_variance = self.nroy_implausibility = self.nroy_y_variance = np.empty(0)
        self.x_scaler = self.y_scaler = None

        # Set plot shapes
        if self.n_y < plot_cols:
            plot_cols = self.n_y
        self.plot_shape_y = [int(np.ceil(self.n_y / plot_cols)), plot_cols]
        if self.n_x < plot_cols:
            plot_cols = self.n_x
        self.plot_shape_x = [int(np.ceil(self.n_x / plot_cols)), plot_cols]

    def clean_up(self, file_types=("hdf5", "npy")):
        """Remove all files and directories created by this wave"""
        clean_up(self.dir, file_types=file_types)

    def surf(self, sensitivity=False, export_csv=False, export_pickle=True, print_summary=False, pickle_name="wave"):
        """Wrapper function to run all the functions required to complete a full wave. Exports wave as pickle"""

        self.gpe_training_validation(print_summary=print_summary)
        self.gpe_emulate()
        self.find_nroy()
        if self.y_lims is not None:
            self.remove_non_physiological()
        self.draw_posterior()

        if sensitivity:
            self.sobol_sensitivity()
        if export_csv:
            self.gpe_export()
        if export_pickle:
            self.pickle_wave(pickle_name=pickle_name)

    def surf_sensitivity(self, export_csv=False, export_pickle=True, print_summary=False, pickle_name="wave"):
        """Wrapper function to run all the functions required for global sensitivity. Exports wave as pickle"""

        self.gpe_training_validation(print_summary=print_summary)
        self.sobol_sensitivity()
        if export_csv:
            self.gpe_export()
        if export_pickle:
            self.pickle_wave(pickle_name=pickle_name)

    def is_flooded(self, i_wave, max_waves=8, min_waves=6, flooded=0.1):
        """Check for convergence of Bayesian History Matching algorithm based on percentage change of NROY space"""

        # Change in NROY size as percentage of the input emulation space (the NROY region of the previous wave)
        sea_level_change = (self.nroy.shape[0] - self.n_emu_init) / self.n_emu_init
        not_flooded = False

        # Abort history matching if maximum amount of waves has been reached
        if (i_wave + 1) >= max_waves:
            not_flooded = False
            if self.save_log:
                update_log(self.log_file, "NROY size change at " + self.name + " is " + f"{sea_level_change:.2f}" +
                           ": history matching terminated because maximum allowed number of waves has been reached",
                           self.print_log)

        # Convergence if NROY region size change is lower than the threshold "flooded"
        elif np.abs(sea_level_change) <= flooded:
            # But only abort if minimum number of waves has not yet been reached
            if i_wave + 1 >= min_waves:
                not_flooded = False
                if self.save_log:
                    update_log(self.log_file,
                               "NROY size change is " + f"{sea_level_change:.2f}" + ": convergence established",
                               self.print_log)
            else:
                not_flooded = True
                if self.save_log:
                    update_log(self.log_file, "NROY size change is " + f"{sea_level_change:.2f}" +
                               ": convergence established but minimum number of waves (" + str(int(min_waves))
                               + ") not yet reached", self.print_log)

        # Otherwise, continue with next wave
        elif np.abs(sea_level_change) > flooded:
            not_flooded = True
            if self.save_log:
                update_log(self.log_file, "NROY size change at " + self.name + " is " + f"{sea_level_change:.2f}"
                           + ": no convergence", self.print_log)

        return not_flooded

    def import_data(self):
        """Extract data from Pandas dataframe"""

        # Names of observed data, list of all indices
        y_names_unique = self.data_df.index.values.tolist()

        # If limits are specified, use them
        if "min" and "max" in self.data_df.columns:
            y_limits = np.array([self.data_df["min"], self.data_df["max"]]).astype('float')
            y_lims = np.empty((2, 0))
        else:
            y_lims = None

        # If categories are specified, use them
        if "category" in self.data_df.columns:
            y_categories = self.data_df["category"].tolist()
            y_cats = []
        else:
            y_cats = None

        # Get labels only
        labels = [i for i in self.data_df.columns if "mu" in i]

        y_names = []
        y_observed = []
        sigma_observed = []

        # Extract data for each label
        for label in labels:

            # If there is a label classifier, find it
            if "_" in label:
                addon = "_" + label.split("_")[1]
            else:
                addon = ""

            # Get names, observed data and standard deviation for this label
            y_names_label = [name + addon for name in self.data_df.index.values.tolist()]

            # If label name is _baseline, remove this substring from the list of names
            if addon == "_baseline":
                y_names_label = [name.replace("_baseline", "") for name in y_names_label]

            y_observed_label = np.array(self.data_df["mu" + addon]).astype('float')
            sigma_observed_label = np.array(self.data_df["sigma" + addon]).astype('float')

            # Add limits
            if y_lims is not None:
                y_lims = np.hstack((y_lims, y_limits))

            # Add categories
            if y_cats is not None:
                y_cats = y_cats.extend(y_categories)

            # Add to full list
            y_names.extend(y_names_label)
            y_observed = np.hstack((y_observed, y_observed_label))
            sigma_observed = np.hstack((sigma_observed, sigma_observed_label))

        # Remove NaNs if present
        i_nan = np.isnan(y_observed)
        if np.any(i_nan):
            y_observed = y_observed[~i_nan]
            sigma_observed = sigma_observed[~i_nan]
            y_names = [y_names[i] for i in range(len(y_names)) if not i_nan[i]]
            if y_lims is not None:
                y_lims = y_lims[:, ~i_nan]
            if y_cats is not None:
                y_cats = [y_cats[i] for i in range(len(y_cats)) if not i_nan[i]]

        return y_names, y_observed, sigma_observed, y_lims, y_cats

    def load_prior(self, prior_dir):
        """Load pickle of final wave of prior"""

        # Load prior pickle
        prior_pickles = sort_nicely(glob.glob(str(prior_dir) + '/*/*.pkl'))
        if len(prior_pickles) == 0:
            raise ValueError("No wave pickles found in prior directory " + str(prior_dir) + ". Please check directory.")

        # Only keep directories that contain "Wave"
        prior_pickles = [pickle for pickle in prior_pickles if "Wave" in pickle]

        # Open last pickle and extract NROY and their names
        with open(prior_pickles[-1], "rb") as f:
            return pickle.load(f)

    def add_prior(self, prior_dir, prior_names=None):
        """Add prior samples to emulation cloud"""

        # Load prior pickle
        prior = self.load_prior(prior_dir)

        # If specific prior parameter are provided, draw only those
        if prior_names is not None:
            i_prior = [prior.x_names.index(par) for par in prior_names]
        else:
            i_prior = list(range(prior.n_x))

        # Draw n_emu random rows from prior NROY
        i_choice = np.random.choice(prior.nroy.shape[0], self.x_emu.shape[0])
        prior_nroy = prior.nroy[i_choice, :]

        # Only add data if this is the first wave
        if self.wave_number == 0:
            # Add to emulation cloud
            self.x_emu = np.hstack((self.x_emu, prior_nroy[:, i_prior]))

            # Also add to NROY if it exists
            if self.nroy is not None:
                i_choice = np.random.choice(prior.nroy.shape[0], self.nroy.shape[0])
                prior_nroy = prior.nroy[i_choice, :]
                self.nroy = np.hstack((self.nroy, prior_nroy[:, i_prior]))

        # Add labels
        self.x_names.extend([prior.x_names[i] for i in i_prior])

        # Add limits using min and max of prior
        self.x_limits = np.hstack((self.x_limits, np.array([np.min(prior.nroy[:, i_prior], axis=0),
                                                            np.max(prior.nroy[:, i_prior], axis=0)])))

        # Add categories
        #if prior.x_cats is not None:
        #    self.x_cats.extend([prior.x_cats[i] for i in i_prior])

        # Add target
        if prior.x_target is not None:
            self.x_target.extend([prior.x_target[i] for i in i_prior])

        # Keep track of which are priors
        self.prior_names = [prior.x_names[i] for i in i_prior]

        # Update n_x
        self.n_x = len(self.x_names)

        # Add priors to log
        if self.wave_number == 0:
            prior_names_str = ""
            for i in range(len(self.prior_names)):
                prior_names_str += self.prior_names[i] + ", "
            if self.save_log:
                update_log(self.log_file,
                           "Added priors to emulation point cloud: " + prior_names_str[:-2], self.print_log)

    def set_scales(self):
        """Fit estimators to normalize x and y using the simulation data, which will span the full parameter space"""
        self.x_scaler = preprocessing.MinMaxScaler()
        self.y_scaler = preprocessing.MinMaxScaler()
        self.x_scaler.fit(self.x_limits)
        self.y_scaler.fit(self.y_sim)

    def split_dataset(self, fraction_training):
        """ Split indices randomly into training and validation sets. Default is 80/20 training/validation split """

        # Check if simulation parameters or data is one-dimensional, if so add dimension
        if self.x_sim.ndim == 1:
            self.x_sim = np.expand_dims(self.x_sim, axis=1)
        if self.y_sim.ndim == 1:
            self.y_sim = np.expand_dims(self.y_sim, axis=1)

        # Shuffle indices
        i_shuffle = np.arange(self.x_sim.shape[0])
        np.random.shuffle(i_shuffle)

        # Split up according to fraction_training
        n_training = int(np.floor(fraction_training * self.x_sim.shape[0]))
        i_training = i_shuffle[:n_training]
        i_validation = i_shuffle[n_training:]

        self.x_training = self.x_sim[i_training, :]
        self.y_training = self.y_sim[i_training, :]
        self.x_validation = self.x_sim[i_validation, :]
        self.y_validation = self.y_sim[i_validation, :]

    def gpe_training_validation(self, fraction_training=0.80, print_summary=None, n_fits=1, learning_rate=0.1,
                                var_lim=(0.01, 2.0), length_lim=(0.01, 2.0), n_training_iter=100):
        """
        Train and validate a GPE model for each observable output y using known x and y. Default of training/validation
        is 80/20 split (specified by fraction_training). All GPE models all stored in Wave.gpe_models.
        """

        if self.save_log:
            update_log(self.log_file, "Training GPEs using PyTorch" + "...",
                       self.print_log if print_summary is None else print_summary)

        # Split into training and validation sets, default is 80/20 split
        self.split_dataset(fraction_training)

        # Scale data and put in PyTorch tensors
        self.set_scales()
        x_training = torch.tensor(scale(self.x_training, self.x_scaler))
        y_training = torch.tensor(scale(self.y_training, self.y_scaler))
        x_validation = torch.tensor(scale(self.x_validation, self.x_scaler))
        y_validation = torch.tensor(scale(self.y_validation, self.y_scaler))

        # Pre-allocate r2 and prediction arrays
        self.r_2 = []
        learning_curves = []
        y_prediction = np.zeros(y_validation.shape)
        y_prediction_variance = np.zeros(y_validation.shape)

        # Train GPE for each output variable y
        for i_y in range(self.n_y):

            # Run n_fits with different initial guesses and keep best fit
            gpe_models = []
            likelihoods = []
            learning_curves_y = []
            losses = np.array([])

            for i_fit in range(n_fits):
                # Initialize likelihood and model
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                model = ExactGPModel(x_training, y_training[:, i_y], likelihood)

                # Find optimal model hyperparameters
                model.train()
                likelihood.train()

                # Use the adam optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Includes GaussianLikelihood parameters

                # "Loss" for GPs - the marginal log likelihood
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

                # Assign random values to parameters if multiple fitting rounds are used
                if n_fits > 1:
                    model.covar_module.base_kernel.lengthscale = random.uniform(length_lim[0], length_lim[1])
                    model.likelihood.noise = random.uniform(var_lim[0], var_lim[1])

                learning_curve = []
                for i in range(n_training_iter):
                    # Zero gradients from previous iteration
                    optimizer.zero_grad()
                    # Output from model
                    output = model(x_training)
                    # Calc loss and backprop gradients
                    loss = -mll(output, y_training[:, i_y])
                    if print_summary:
                        print('Iter %d/%d - Loss: %.3f' % (i + 1, n_training_iter, loss.item()))
                    learning_curve.append(loss.item())
                    loss.backward()
                    optimizer.step()

                # Add to collection
                gpe_models.append(model)
                likelihoods.append(likelihood)
                losses = np.append(losses, loss.item())
                learning_curves_y.append(learning_curve)

            # Select best fit
            gpe_model = gpe_models[np.argmin(losses)].double()
            likelihood = likelihoods[np.argmin(losses)].double()
            learning_curves.append(learning_curves_y[np.argmin(losses)])

            # Save model state and training data
            self.gpe_models.append(gpe_model)
            self.gpe_likelihoods.append(likelihood)
            self.save_emulator(gpe_model, likelihood, self.dir_gpe / str("gpe_" + self.y_names[i_y] + ".pt"))

            # Predict outcomes for validation
            gpe_model.eval()
            likelihood.eval()

            with torch.no_grad():
                # Predict mean and variance
                observed_pred = likelihood(gpe_model(x_validation.double()))
                y_prediction[:, i_y] = observed_pred.mean.numpy()
                y_prediction_variance[:, i_y] = observed_pred.variance.numpy()

                # Compute R2
                self.r_2.append(metrics.r2_score(y_validation[:, i_y], y_prediction[:, i_y]))

        # Wrap-up: unscale predictions to real value and plot R2
        self.y_prediction = unscale(y_prediction, self.y_scaler)
        self.y_prediction_variance = y_prediction_variance / self.y_scaler.scale_ ** 2
        self.plot_r2()
        self.plot_r2(confidence=True, fig_name="r2_confidence.pdf")
        self.plot_learning(learning_curves)

    def gpe_emulate(self, print_log=None):
        """
        Emulate the full parameter space and calculate implausibility score of each emulation
        """

        if self.save_log:
            update_log(self.log_file, "Emulating " + str(self.n_emu) + " points...",
                       self.print_log if print_log is None else print_log)

        if len(self.gpe_models) > 0:
            gpe_models = self.gpe_models
            gpe_likelihoods = self.gpe_likelihoods
        else:
            # Load the GPE models from file
            gpe_models = []
            gpe_likelihoods = []
            for i_y in range(self.n_y):
                model, likelihood, x_scaler, y_scaler = self.load_emulator(
                    self.dir_gpe / str("gpe_" + self.y_names[i_y] + ".pt"))
                gpe_models.append(model)
                gpe_likelihoods.append(likelihood)
                # Set scalers if previously stored
                if x_scaler is not None:
                    self.x_scaler = x_scaler
                if y_scaler is not None:
                    self.y_scaler = y_scaler

        # Scale input x to normalized values
        x_emu = torch.tensor(scale(self.x_emu, self.x_scaler), dtype=torch.float32)
        self.y_emu = np.zeros((x_emu.shape[0], self.n_y))
        self.y_emu_variance = np.zeros((x_emu.shape[0], self.n_y))

        # Record time
        start_time = time.time()

        # Emulate each y
        for i_y in range(self.n_y):
            # Initialize likelihood and model
            likelihood = gpe_likelihoods[i_y]
            gpe_model = gpe_models[i_y]

            # Evaluation mode
            gpe_model.eval()
            likelihood.eval()

            # Predict mean and variance
            with torch.no_grad():
                observed_pred = likelihood(gpe_model(x_emu.double()))
                self.y_emu[:, i_y] = observed_pred.mean.numpy()
                self.y_emu_variance[:, i_y] = observed_pred.variance.numpy()

        # Report time to completion
        if self.save_log:
            update_log(self.log_file, "Emulation completed in " + str(round(time.time() - start_time, 2)) + " seconds",
                       self.print_log)

        # Unscale emulated y, variance scales quadratically
        self.y_emu = unscale(self.y_emu, self.y_scaler)
        self.y_emu_variance = self.y_emu_variance / self.y_scaler.scale_ ** 2

    def gpe_training_validation_multitask(self, fraction_training=0.80, print_summary=None, learning_rate=0.1,
                                          rank=2, n_training_iter=100, gpe_name="gpe_multitask.pt"):
        """
        Train and validate a multitask GPE model for all observable outputs y using known x and y.
        Default training/validation is 80/20 split (specified by fraction_training).
        """

        if self.save_log:
            update_log(self.log_file, "Training multitask GPE using PyTorch...",
                       self.print_log if print_summary is None else print_summary)

        # Split into training and validation sets
        self.split_dataset(fraction_training)

        # Scale data and convert to PyTorch tensors
        self.set_scales()
        x_training = torch.tensor(scale(self.x_training, self.x_scaler), dtype=torch.float32)
        y_training = torch.tensor(scale(self.y_training, self.y_scaler), dtype=torch.float32)
        x_validation = torch.tensor(scale(self.x_validation, self.x_scaler), dtype=torch.float32)
        y_validation = torch.tensor(scale(self.y_validation, self.y_scaler), dtype=torch.float32)

        # Initialize likelihood and model
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.n_y)
        model = MultitaskGPModel(x_training, y_training, likelihood, num_tasks=self.n_y, rank=rank)

        # Training mode
        model.train()
        likelihood.train()

        # Use Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # Training loop
        learning_curve = []
        for i in range(n_training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_training)
            # Calculate loss and backpropagate gradients
            loss = -mll(output, y_training)
            if print_summary:
                print(f'Iter {i + 1}/{n_training_iter} - Loss: {loss.item():.3f}')
            learning_curve.append(loss.item())
            loss.backward()
            optimizer.step()

        # Save model, likelihood, training data, and metadata including scalers
        self.save_emulator(model, likelihood, self.dir_gpe / gpe_name)
        self.gpe_models.append(model)
        self.gpe_likelihoods.append(likelihood)

        # Evaluation mode
        model.eval()
        likelihood.eval()

        # Pre-allocate R² and prediction arrays
        self.r_2 = []
        y_prediction = np.zeros(y_validation.shape)
        y_prediction_variance = np.zeros(y_validation.shape)

        # Predict validation results
        with torch.no_grad():
            observed_pred = likelihood(model(x_validation))
            y_prediction = observed_pred.mean.numpy()
            y_prediction_variance = observed_pred.variance.numpy()

        # Compute R² for each output
        for i_y in range(self.n_y):
            self.r_2.append(metrics.r2_score(y_validation[:, i_y], y_prediction[:, i_y]))

        # Wrap-up: unscale predictions to real value and plot R2
        self.y_prediction = unscale(y_prediction, self.y_scaler)
        self.y_prediction_variance = y_prediction_variance / self.y_scaler.scale_ ** 2
        self.plot_r2()
        self.plot_r2(confidence=True, fig_name="r2_confidence.pdf")
        self.plot_learning([learning_curve])

        return model, likelihood

    def gpe_emulate_multitask(self, print_log=None, gpe_name="gpe_multitask.pt", batch_size=10000, fast_mode=False):
        """
        Emulate the full parameter space using a multitask GPE model and calculate
        implausibility score of each emulation
        """

        if self.save_log:
            update_log(self.log_file, "Emulating " + str(self.x_emu.shape[0]) + " points with multitask GPE...",
                       self.print_log if print_log is None else print_log)

        # Record time
        start_time = time.time()

        # Load model, likelihood, training data, and metadata including scalers
        if len(self.gpe_models) > 0:
            model = self.gpe_models[0]
            likelihood = self.gpe_likelihoods[0]
        else:
            # Load the GPE model from file
            model, likelihood, x_scaler, y_scaler = self.load_emulator(self.dir_gpe / gpe_name)
            # Set scalers from loaded model if available
            if x_scaler is not None:
                self.x_scaler = x_scaler
            if y_scaler is not None:
                self.y_scaler = y_scaler

        x_emu = torch.tensor(scale(self.x_emu, self.x_scaler), dtype=torch.float32)
        self.y_emu = np.zeros((x_emu.shape[0], self.n_y))
        self.y_emu_variance = np.zeros((x_emu.shape[0], self.n_y))

        # Process in batches to avoid memory issues
        batch_size = min(batch_size, x_emu.shape[0])
        num_batches = (x_emu.shape[0] + batch_size - 1) // batch_size

        # Evaluation mode
        model.eval()
        likelihood.eval()

        if fast_mode:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, x_emu.shape[0])

                    current_batch = x_emu[start_idx:end_idx]

                    # Make predictions
                    observed_pred = likelihood(model(current_batch))

                    # Store results
                    self.y_emu[start_idx:end_idx] = observed_pred.mean.numpy()
                    self.y_emu_variance[start_idx:end_idx] = observed_pred.variance.numpy()
        else:
            with torch.no_grad():
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, x_emu.shape[0])

                    current_batch = x_emu[start_idx:end_idx]

                    # Make predictions
                    observed_pred = likelihood(model(current_batch))

                    # Store results
                    self.y_emu[start_idx:end_idx] = observed_pred.mean.numpy()
                    self.y_emu_variance[start_idx:end_idx] = observed_pred.variance.numpy()


        # Report time to completion
        if self.save_log:
            update_log(self.log_file,
                       "Multitask emulation completed in " + str(round(time.time() - start_time, 2)) + " seconds",
                       self.print_log if print_log is None else print_log)

        # Unscale emulated y
        self.y_emu = unscale(self.y_emu, self.y_scaler)
        self.y_emu_variance = self.y_emu_variance / self.y_scaler.scale_ ** 2

        return model, likelihood

    def save_emulator(self, model, likelihood, file_path, print_save=False):
        """Save a trained multitask GP emulator to a file
        """
        # Create a dictionary with all components needed to restore the emulator
        state_dict = {
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
            'train_inputs': model.train_inputs[0],  # Store training data
            'train_targets': model.train_targets,
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler,
        }

        # If model.num_tasks exists, add it to the dictionary
        if hasattr(model, 'num_tasks'):
            state_dict['num_tasks'] = model.num_tasks
        if hasattr(model, 'covar'):
            state_dict['rank'] = model.covar_module.task_covar_module.covar_factor.shape[-1],

        # Save to file
        torch.save(state_dict, file_path)
        if print_save:
            print(f"Emulator saved to {file_path}")

    def load_emulator(self, file_path):
        """Load a trained multitask GP emulator from a file"""
        # Load the state dictionary from file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            state_dict = torch.load(file_path)

        # Extract the training data and model parameters
        train_x = state_dict['train_inputs']
        train_y = state_dict['train_targets']
        num_tasks = state_dict['num_tasks']
        rank = state_dict.get('rank', 2)  # Default to 2 if not stored
        x_scaler = state_dict.get('x_scaler', None) # Use the stored x_scaler if available, otherwise None
        y_scaler = state_dict.get('y_scaler', None) # Use the stored y_scaler if available, otherwise None

        # Create a new likelihood
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

        # Create a new model with the same architecture
        model = MultitaskGPModel(train_x, train_y, likelihood, num_tasks, rank)

        # Load the saved state dictionaries
        model.load_state_dict(state_dict['model_state_dict'])
        likelihood.load_state_dict(state_dict['likelihood_state_dict'])

        # Set to evaluation mode
        model.eval()
        likelihood.eval()

        print(f"Emulator loaded from {file_path}")
        return model, likelihood, x_scaler, y_scaler

    def find_nroy(self, plot_nroy=True):
        """Use the implausibility score to identify the NROY region for this wave"""

        # Calculate implausibility of each emulated y
        self.implausibility_y = np.sqrt((self.y_emu - self.y_observed) ** 2.0 /
                                        (self.y_emu_variance + self.sigma_observed ** 2.0))

        # Compile overall implausibility score: take maximum of all y scores for "worst-case scenario"
        self.implausibility = np.max(self.implausibility_y, axis=1)

        # Identify not-ruled out yet region: true if not implausible, false if implausible according to the threshold value
        self.i_nroy = self.implausibility <= self.threshold
        self.nroy = self.x_emu[self.i_nroy, :]
        self.nroy_y = self.y_emu[self.i_nroy, :]
        self.nroy_y_variance = self.y_emu_variance[self.i_nroy, :]
        self.nroy_implausibility = self.implausibility[self.i_nroy]

        if self.save_log:
            update_log(self.log_file, str(int(self.nroy.shape[0])) +
                       " points remaining in NROY region (" + str(round(self.nroy.shape[0] / self.n_emu0 * 100, 2)) +
                       "% of original space)", self.print_log)

        # Find y that is the most implausible
        i_max = np.argmax(self.implausibility_y, axis=1)
        i_max, counts = np.unique(i_max, return_counts=True)
        i_max = i_max[np.argmax(counts)]

        if self.save_log:
            update_log(self.log_file, str(self.y_names[i_max]) + " is the most implausible output ("
                       + str(round(counts[np.argmax(counts)] / np.sum(counts) * 100, 2)) + "% of all points)",
                       self.print_log)

        # Plot NROY region
        if plot_nroy:
            self.plot_nroy()

    def remove_non_physiological(self):
        """Remove non-physiological points from NROY region"""

        # Get indices of non-physiological points, i.e. points outside user-defined limits for y
        i_lower = np.any(self.nroy_y < self.y_lims[0, :], axis=1)
        i_upper = np.any(self.nroy_y > self.y_lims[1, :], axis=1)
        i_remove = np.any(np.vstack((i_lower, i_upper)), axis=0)

        # Remove rows from nroy and nroy_y
        self.nroy = np.delete(self.nroy, i_remove, axis=0)
        self.nroy_y = np.delete(self.nroy_y, i_remove, axis=0)
        self.nroy_y_variance = np.delete(self.nroy_y_variance, i_remove, axis=0)
        self.nroy_implausibility = np.delete(self.nroy_implausibility, i_remove, axis=0)

        if self.save_log:
            update_log(self.log_file, str(int(self.nroy.shape[0])) + " points remaining in NROY region after limits check",
                       self.print_log)

    def gpe_export(self, filename="emulate_results"):
        """Export emulated results and their input parameters alongside implausibility score and threshold satisfaction"""

        # Collect all data in one matrix
        data_out = np.concatenate((self.x_emu, self.y_emu, self.y_emu_variance, self.implausibility_y), axis=1)
        data_out = np.vstack((data_out.T, self.implausibility, self.i_nroy)).T

        # Collect all labels in one list
        y_labels_exp = [s + ' expected' for s in list(self.y_names)]
        y_labels_var = [s + ' variance' for s in list(self.y_names)]
        y_labels_implausible = [s + ' implausibility' for s in list(self.y_names)]
        labels_out = list(self.x_names) + y_labels_exp + y_labels_var + y_labels_implausible + ["Implausibility",
                                                                                                "NROY"]

        # Create dataframe and export
        df = pd.DataFrame(data_out, columns=labels_out)
        df.to_csv(self.dir / str(filename + ".csv"))

    def pickle_wave(self, pickle_dir=None, pickle_name="wave"):
        """Pickly wave"""

        if pickle_dir is None:
            pickle_dir = self.dir

        # Clear directories for cross-platform compatibility
        if self.os_compatibility:
            self.dir = None
            self.dir_sim = None
            self.gpe_models = None
            self.likelihoods = None

        # Pickly wave
        with open(pickle_dir / str(pickle_name + ".pkl"), 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def regenerate_emu(self, factor=0.1):
        """ Use the "Cloud method" from Coveney & Clayton (2018) to generate additional points in the next NROY wave
        input space. Iterate through every point in the current NROY space and add a point that is located within scale_std
        times the standard deviation of the current distribution from the point that is iterated on. The iteration continues
        until the number of points in the NROY reaches n_emu"""

        # Get boundaries of NROY region
        nroy_min = np.min(self.nroy, axis=0)
        nroy_max = np.max(self.nroy, axis=0)
        scale_nroy = factor * (nroy_max - nroy_min)

        # Repeat adding points until x_emu is filled up with n_emulate points again
        while self.x_emu.shape[0] < self.n_emu_min:

            # Create one random point for every existing point within the NROY region, points can be "factor" outside the
            # original NROY region before resampling
            x_temp = np.random.normal(loc=self.nroy, scale=scale_nroy)

            # Regenerate points that are not within the set parameter boundaries
            repeat = True
            attempts = 0
            while repeat and attempts < 10:
                # Find points that were not within the parameter bounds
                i_min = np.any(x_temp <= self.x_limits[0, :], axis=1)
                i_max = np.any(x_temp >= self.x_limits[1, :], axis=1)
                # i_min = np.any(x_temp <= nroy_min, axis=1)
                # i_max = np.any(x_temp >= nroy_max, axis=1)
                i_outside = np.where(np.any(np.vstack((i_min, i_max)).T, axis=1))[0]

                # Stop if all points are inside the boundaries
                if np.all(i_outside) is False:
                    repeat = False
                # Regenerate points that were outside the boundaries
                else:
                    # Get new estimates
                    x_temp_regen = np.random.normal(loc=x_temp[i_outside, :], scale=scale_nroy)

                    # Only assign back to temp if it is within the set bounds
                    i_min = np.all(x_temp_regen >= self.x_limits[0, :], axis=1)
                    i_max = np.all(x_temp_regen <= self.x_limits[1, :], axis=1)
                    # i_min = np.all(x_temp_regen >= nroy_min, axis=1)
                    # i_max = np.all(x_temp_regen <= nroy_max, axis=1)
                    i_inside = np.all(np.vstack((i_min, i_max)).T, axis=1)
                    x_temp[i_outside[i_inside], :] = x_temp_regen[i_inside, :]

                    # Only take 10 attempts, then remove points entirely
                    attempts += 1
                    if attempts == 10:
                        x_temp = np.delete(x_temp, i_outside[~i_inside], axis=0)

            # When we get close to the required number of points, too many points will be added. Prevent this by randomly
            # removing the excess number of points from the current temp array
            if (self.n_emu_min - self.x_emu.shape[0]) < x_temp.shape[0]:
                idx = np.random.randint(x_temp.shape[0], size=self.n_emu_min - self.x_emu.shape[0])
                x_temp = x_temp[idx, :]

            # Add new points to the emulation space for this wave
            self.x_emu = np.concatenate((self.x_emu, x_temp))

    def draw_posterior(self, nroy=None, n_samples=None):
        """Randomly sample x_sim from the NROY region to obtain the posterior distribution. Optional: NROY region can be
        specified form outside."""
        if not np.any(nroy):
            nroy = self.nroy
        if not n_samples:
            n_samples = self.n_posterior

        # Randomly draw n_sim points from the NROY region
        idx = np.random.randint(nroy.shape[0], size=n_samples)
        self.x_posterior = nroy[idx, :]

    def plot_results(self):
        """Collection of all plot functions for a single wave"""
        self.plot_r2()
        self.sobol_sensitivity()
        # self.plot_implausibility()        # Disabled by default since it is very time-consuming

    def plot_r2(self, fig_name="r2.pdf", confidence=False, save_fig=True, show_fig=False):
        """Plot R2 and Y observed vs Y predicted"""

        sns.set_theme(style="white")
        fig = plt.figure(figsize=(2.5 * self.plot_shape_y[1], 2.5 * self.plot_shape_y[0]))

        # Plot R2 ald correlaiton for every y variable
        for i_y in range(self.n_y):
            ax = fig.add_subplot(self.plot_shape_y[0], self.plot_shape_y[1], i_y + 1)

            # Axes limits
            y_min = np.minimum(np.min(self.y_validation[:, i_y]), np.min(self.y_prediction[:, i_y]))
            y_max = np.maximum(np.max(self.y_validation[:, i_y]), np.max(self.y_prediction[:, i_y]))
            centroid = 0.5 * y_min + 0.5 * y_max
            y_l = y_min - 0.2 * centroid
            y_u = y_max + 0.2 * centroid

            ax.set_title(self.y_names[i_y])

            # Diagonal
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color=(58 / 255, 133 / 255, 140 / 255), zorder=1,
                    linewidth=4)

            # Data points, sort on ascending validation y
            i_sort = np.argsort(self.y_validation[:, i_y])

            if confidence:
                ax.errorbar(self.y_validation[i_sort, i_y], self.y_prediction[i_sort, i_y],
                            yerr=np.sqrt(self.y_prediction_variance[i_sort, i_y]) * 2, fmt='o', color='black',
                            ecolor=(243 / 255, 177 / 255, 93 / 255), elinewidth=2, alpha=0.5, capsize=0)

                # ax.fill_between(self.y_validation[i_sort, i_y],
                #                 self.y_prediction[i_sort, i_y] - np.sqrt(self.y_prediction_variance[i_sort, i_y])*2,
                #                 self.y_prediction[i_sort, i_y] + np.sqrt(self.y_prediction_variance[i_sort, i_y])*2,
                #                 color=(243/255, 177/255, 93/255), alpha=0.5, zorder=0)

            ax.scatter(self.y_validation[i_sort, i_y], self.y_prediction[i_sort, i_y],
                       color=(243 / 255, 177 / 255, 93 / 255), s=50, zorder=10, edgecolor='k')

            # Show R-squared value
            ax.annotate(f"R$^2$ = {self.r_2[i_y]:0.2f}", xycoords='axes fraction', xy=(0.15, 0.8), size='medium',
                        color='black')

            ax.set(xlim=(y_l, y_u), ylim=(y_l, y_u), xticks=(y_l, y_u), yticks=(y_u,))
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.set_aspect('equal', 'box')
            # if i_y == 0:
            #     ax.set(xlabel='CardioGrowth', ylabel='GPE')
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2)

        plt.tight_layout()
        if save_fig:
            plt.savefig(self.dir / fig_name, bbox_inches='tight')
        if show_fig:
            plt.show()
        plt.close()

    def plot_learning(self, learning_curves, fig_name="learningcurves.pdf"):
        """Plot GPE learning curves"""

        x = np.arange(len(learning_curves[0]))

        fig = plt.figure(figsize=(2.5 * self.plot_shape_y[1], 2.5 * self.plot_shape_y[0]))

        # Plot R2 ald correlation for every y variable
        for i_y in range(len(learning_curves)):
            ax = fig.add_subplot(self.plot_shape_y[0], self.plot_shape_y[1], i_y + 1)

            ax.set_title(self.y_names[i_y])

            # Data points
            ax.plot(x, learning_curves[i_y], color=(58 / 255, 133 / 255, 140 / 255), zorder=10, linewidth=3)

            ax.set(xlim=(x[0], x[-1]), xticks=(x[0], np.ceil(x[-1] / 2), x[-1] + 1))
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.set_box_aspect(1)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2)

        plt.tight_layout()
        plt.savefig(self.dir / fig_name, bbox_inches='tight')
        plt.close()

    def plot_nroy(self, fig_name="nroy.pdf", show_fig=False, save_fig=True):

        self.plot_nroy_kdes(self.x_names, self.plot_shape_x, self.x_emu, self.nroy, observed=self.x_target,
                            fig_name=fig_name.split(".")[0] + "_x." + fig_name.split(".")[1],
                            show_fig=show_fig, save_fig=save_fig)

        self.plot_nroy_kdes(self.y_names, self.plot_shape_y, self.y_emu, self.nroy_y,
                            observed=self.y_observed, sigma_observed=self.sigma_observed,
                            fig_name=fig_name.split(".")[0] + "_y." + fig_name.split(".")[1],
                            show_fig=show_fig, save_fig=save_fig)

    def plot_nroy_kdes(self, names, shape, emu, nroy, observed=None, sigma_observed=None,
                       fig_name="nroy.pdf", show_fig=False, save_fig=True):
        """Plot NROY region y values"""

        colors = sns.color_palette("cubehelix", len(names))
        legend_labels = ['Emulated', 'NROY']
        if observed is not None:
            legend_labels.append('Observed')
            if sigma_observed is not None:
                legend_labels.append(r"$2\sigma$")

        fig = plt.figure(figsize=(2.5 * shape[1], 2.5 * shape[0]))

        for i in range(len(names)):
            ax = fig.add_subplot(shape[0], shape[1], i + 1)
            ax.set_title(names[i])
            sns.kdeplot(data=emu[:, i], ax=ax, fill=True, color=colors[i], alpha=0.2)
            sns.kdeplot(data=nroy[:, i], ax=ax, fill=True, color=colors[i], alpha=0.8)
            if observed is not None:
                ax.axvline(x=observed[i], color='k', linestyle='-')
                if sigma_observed is not None:
                    ax.axvline(x=observed[i] - 2 * sigma_observed[i], color='k', linestyle='--')
                    ax.axvline(x=observed[i] + 2 * sigma_observed[i], color='k', linestyle='--')

            # Only set ylabel in the leftmost column
            if i % shape[1] == 0:
                ax.set(ylabel='Probability density')
            else:
                ax.set(ylabel="")

        # Set legend
        fig.legend(labels=legend_labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.4, 1.1))

        plt.tight_layout()
        if save_fig:
            plt.savefig(self.dir / fig_name, bbox_inches='tight')
        if not show_fig:
            plt.close()

    def sobol_sensitivity(self, n_samples=2 ** 15):
        """Perform Sobol sensitivity analysis this wave's trained GPEs and parameter space"""

        if self.save_log:
            update_log(self.log_file, "Performing sensitivity analysis" + "...", self.print_log)

        # Define problem: range and dimensions
        self.sobol_problem = {
            'num_vars': self.n_x,
            'names': self.x_names,
            'bounds': self.x_limits.T,
            'outputs': self.y_names
        }

        # Sample space
        x = saltelli.sample(self.sobol_problem, n_samples)

        # Evaluate (save a copy of the current emulations and restore after analysis)
        x_emu, y_emu, y_emu_variance = np.copy(self.x_emu), np.copy(self.y_emu), np.copy(self.y_emu_variance)
        self.x_emu = x
        self.gpe_emulate()

        # Analyze
        self.sobol_indices = [sobol.analyze(self.sobol_problem, y) for y in self.y_emu.T]

        # Restore original emulations to Wave class and clear from sensitivity analysis to save storage space
        self.x_emu, self.y_emu, self.y_emu_variance = x_emu, y_emu, y_emu_variance

    def plot_implausibility(self, fig_name="implausibility.png"):
        """Plot implausibility for all variables"""

        # Convert x to Pandas dataframe and add implausibility as a column
        df = pd.DataFrame(self.x_emu, columns=self.x_names)
        df['Implausibility'] = self.implausibility
        df = df.sort_values("Implausibility", ascending=False)

        # Plot implausibility in the multidimensional space
        g = sns.PairGrid(df, hue="Implausibility", corner=True, height=1.25, aspect=1, despine=False)
        g.map(sns.scatterplot, edgecolor=None)
        g.add_legend()
        g.savefig(self.dir / fig_name, bbox_inches='tight', dpi=300)
        plt.close()

