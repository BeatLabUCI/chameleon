import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import glob
from mne_connectivity.viz import plot_connectivity_circle
from .utils import scale, sort_nicely, label, get_cat_colors, normalize_data_ci


class Camouflage:
    """
    Plots results of completed waves of a history matching process. A tide contains a series of waves.
    """

    def __init__(self, root_dir, cmap_waves=None, cmap_y=None, cmap_x=None, dir_output="Results",
                 posterior_label="Posterior", confidence_lim=2.0, wave_pickles=None):

        # Detect all wave results, stored as pickles.
        # Sort by size to ensure human sorting (i.e. 1, 2, ...10, 11 instead of 10, 11, 1, 2...)
        self.root_dir = root_dir
        if wave_pickles is None:
            self.wave_pickles = sort_nicely(glob.glob(str(root_dir) + '/*/*.pkl'))
        else:
            self.wave_pickles = wave_pickles
        if len(self.wave_pickles) == 0:
            raise ValueError("No wave pickles found in " + str(root_dir) + ". Please check directory.")

        # Remove posterior
        if posterior_label is not None:
            # Find if any wave contains posterior label and remove them
            i_posterior = [i for i, s in enumerate(self.wave_pickles) if posterior_label in s]
            for i in i_posterior[::-1]:
                self.wave_pickles.pop(i)

        # Create directory to store results
        self.dir_output = root_dir / dir_output
        self.dir_output.mkdir(exist_ok=True, parents=True)

        # Unpickle waves and create a tidal list of waves
        self.waves = []
        wave = None
        for wave_pickle in self.wave_pickles:
            pickle_file = open(str(wave_pickle), "rb")
            wave = pickle.load(pickle_file)
            pickle_file.close()
            self.waves.append(wave)

        # Collect basic information
        self.n_waves = len(self.waves)
        self.wave_numbers = np.arange(self.n_waves) + 1
        self.wave_names = [wave.name for wave in self.waves]
        self.n_y = wave.n_y
        self.n_x = wave.n_x
        self.y_observed = wave.y_observed
        self.sigma_observed = wave.sigma_observed
        self.x_limits = wave.x_limits
        self.x_names = wave.x_names
        self.x_cats = wave.x_cats
        self.y_names = wave.y_names
        self.y_cats = wave.y_cats
        self.threshold_final = wave.threshold
        self.x_target = wave.x_target

        # Find wave.x_names that are not in wave.prior_names
        self.x_names_matched = [name for name in self.x_names if name not in wave.prior_names]

        # Default values for color maps and the like
        if cmap_y is None:
            self.cmap_y = sns.color_palette("cubehelix", self.n_y, as_cmap=False)
        else:
            self.cmap_y = cmap_y

        if cmap_x is None:
            self.cmap_x = sns.color_palette("cubehelix", self.n_x, as_cmap=False)
        else:
            self.cmap_x = cmap_x
        self.cmap_x_sensitivity = cmap_x

        if cmap_waves is None:
            self.cmap_waves = sns.cubehelix_palette(self.n_waves+1, as_cmap=False)
        else:
            self.cmap_waves = cmap_waves

        # Get confidence interval limits, use 95% confidence interval as default
        self.confidence_lim = confidence_lim

    def roll(self, lims=None, nroy_full=False, n_samples=10000):
        """Wrapper to run all plot functions with default values"""

        # Use Seaborn to set plot style back to default, in case it was changed by Jupyter
        sns.set_theme(style="white", palette=None)

        self.plot_implausibility_density()
        self.plot_implausibility_counts()
        self.plot_sensitivity_matrix()
        self.plot_sensitivity_matrix()
        self.roll_sensitivity()
        self.plot_nroy_data()
        self.plot_sim_space()
        self.plot_nroy_x(n_samples=n_samples)
        self.plot_nroy_y(n_samples=n_samples, lims=lims)
        self.plot_waves_y(n_samples=n_samples)
        self.plot_waves_x(n_samples=n_samples)

        # Plot full NROY region if desired: slooow
        if nroy_full and n_samples is not None:
            self.plot_nroy_x(filename="nroy_x_full.pdf")
            self.plot_nroy_y(lims=lims, filename="nroy_x_full.pdf")
        # self.plot_nroy_waves()

    def roll_sensitivity(self, show_fig=False):
        """Wrapper to run all plot functions with default values for sensitivity analysis"""
        self.plot_sensitivity_matrix(show_fig=show_fig)
        self.plot_sensitivity_total(show_fig=show_fig)
        self.plot_sensitivity_circles(show_fig=show_fig)
        self.plot_connectivity(show_fig=show_fig)

    def plot_waves_y(self, show_fig=False, n_samples=None):
        """Plot the wave evolution of the NROY y values and the observed mean and 95% confidence interval"""

        # Start with initial
        df_nroy = pd.DataFrame(data=self.waves[0].y_emu, columns=self.y_names)
        df_nroy["Wave"] = "Initial"

        for wave in self.waves:
            df = pd.DataFrame(data=wave.nroy_y, columns=self.y_names)
            df["Wave"] = wave.name
            # Sample data if desired
            if (n_samples is not None) and (n_samples < len(df)):
                df = df.sample(n_samples)
            df_nroy = pd.concat([df_nroy, df])

        for i_y in range(self.n_y):
            self.plot_waves(df_nroy, self.y_names[i_y], filename="waves_y_" + self.y_names[i_y] + ".pdf",
                            mean_observed=self.y_observed[i_y], sigma_observed=self.sigma_observed[i_y],
                            show_fig=show_fig)

    def plot_waves_x(self, show_fig=False, n_samples=None):
        """Plot the wave evolution of the NROY values"""

        # Start with initial
        df_nroy = pd.DataFrame(data=self.waves[0].x_emu, columns=self.x_names)
        df_nroy["Wave"] = "Initial"

        for wave in self.waves:
            df = pd.DataFrame(data=wave.nroy, columns=self.x_names)
            df["Wave"] = wave.name
            # Sample data if desired
            if (n_samples is not None) and (n_samples < len(df)):
                df = df.sample(n_samples)
            df_nroy = pd.concat([df_nroy, df])

        for i_x in range(self.n_x):
            xlim = [self.x_limits[0, i_x] - 0.5*np.mean(self.x_limits[:, i_x]),
                    self.x_limits[1, i_x] + 0.5*np.mean(self.x_limits[:, i_x])]
            if self.x_target is None:
                self.plot_waves(df_nroy, self.x_names[i_x], filename="waves_x_" + self.x_names[i_x] + ".pdf", xlim=xlim,
                                show_fig=show_fig)
            else:
                self.plot_waves(df_nroy, self.x_names[i_x], filename="waves_x_" + self.x_names[i_x] + ".pdf", xlim=xlim,
                                mean_observed=self.x_target[i_x], show_fig=show_fig)

    def plot_waves(self, df, y_name, mean_observed=None, sigma_observed=None, filepath=None, filename="waves.pdf",
                   xlim=None, show_fig=False):
        """ Plot waves as wave plots: distribution of certain column in the pandas data over every wave"""

        if not filepath:
            filepath = self.dir_output

        # Initialize the FacetGrid object
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        g = sns.FacetGrid(df, row="Wave", hue="Wave", aspect=8, height=.58, palette=self.cmap_waves, xlim=xlim)

        if (mean_observed is not None) and (sigma_observed is not None):
            clips = (mean_observed - 4*self.confidence_lim*sigma_observed,
                     mean_observed + 4*self.confidence_lim*sigma_observed)
        else:
            clips = (None, None)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, y_name,
              bw_adjust=.5, clip_on=False,
              fill=True, alpha=1, linewidth=1.5, clip=clips)
        g.map(sns.kdeplot, y_name, clip_on=False, color="w", lw=2, bw_adjust=.5, clip=clips)

        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        # Draw data lines when applicable
        if mean_observed:
            g.refline(x=mean_observed, linewidth=2, linestyle="-", color=[0.5, 0.5, 0.5])
            if sigma_observed:
                g.refline(x=mean_observed - self.confidence_lim*sigma_observed, linewidth=2, linestyle=":", color=[0.5, 0.5, 0.5])
                g.refline(x=mean_observed + self.confidence_lim*sigma_observed, linewidth=2, linestyle=":", color=[0.5, 0.5, 0.5])

        g.map(label, y_name)

        plt.xlabel(y_name)

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.25)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        # Save figure
        plt.savefig(filepath / filename, dpi=300)

        if show_fig:
            plt.show()
        plt.close()
        sns.reset_defaults()

    def plot_variance(self, i_wave=-1, filepath=None, filename="variance"):
        """Plot the multidimensional variance space of a wave of choice (default final wave)"""

        if not filepath:
            filepath = self.dir_output

        # Select wave of choice (default is final wave)
        wave = self.waves[i_wave]

        # Plot variance space for each y variable
        for i_y, y_name in enumerate(self.y_names):
            df = pd.DataFrame(data=wave.x_emu, columns=self.x_names)
            df["Variance"] = wave.y_emu_variance[:, i_y]
            df = df.sort_values("Variance")

            # Plot NROY region evolution over the waves
            g = sns.PairGrid(df, vars=self.x_names, hue="Variance", corner=True, height=1.25, aspect=1,
                             despine=False)
            g.map(sns.scatterplot, edgecolor=None, s=15, alpha=0.1)
            g.add_legend(fontsize="small", title="")
            g._legend.set_bbox_to_anchor((0.85, 0.77))
            g.savefig(filepath / str(filename + "_" + str(y_name) + ".png"), bbox_inches='tight', dpi=300)
            plt.close()

    def plot_mean_difference(self, i_wave=-1, filepath=None, filename="meandiff"):
        """Plot the multidimensional variance space of a wave of choice (default final wave)"""

        if not filepath:
            filepath = self.dir_output

        # Select wave of choice (default is final wave)
        wave = self.waves[i_wave]

        # Plot variance space for each y variable
        for i_y, y_name in enumerate(self.y_names):
            df = pd.DataFrame(data=wave.x_emu, columns=self.x_names)
            df["Variance"] = np.abs(wave.y_emu[:, i_y] - self.y_observed[i_y])
            df = df.sort_values("Variance")

            # Plot NROY region evolution over the waves
            g = sns.PairGrid(df, vars=self.x_names, hue="Variance", corner=True, height=1.25, aspect=1,
                             despine=False)
            g.map(sns.scatterplot, edgecolor=None, s=15)
            g.add_legend(fontsize="small", title="")
            g._legend.set_bbox_to_anchor((0.85, 0.77))
            g.savefig(filepath / str(filename + "_" + str(y_name) + ".png"), bbox_inches='tight', dpi=300)
            plt.close()

    def plot_implausibility_density(self, filepath=None, filename="implausibility_density.pdf"):
        """Plot cumulative distribution of implausibility by wave"""

        if not filepath:
            filepath = self.dir_output

        # Obtain implausibility and wave number of all emulations, collect in single dataframe
        df_implausibility = pd.DataFrame()
        for wave in self.waves:
            df = pd.DataFrame(data=wave.implausibility, columns=["Implausibility"])
            df["Wave"] = wave.name
            df_implausibility = pd.concat([df_implausibility, df])

        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        # Plot colors
        sns.kdeplot(
            data=df_implausibility, x="Implausibility", hue="Wave", fill=True, palette=self.cmap_waves[1:],
            cumulative=True, common_norm=False, common_grid=True, alpha=1, color="w", lw=2
        )
        # Plots contour lines
        plt.setp(ax.get_legend().get_texts(), fontsize="small")
        sns.kdeplot(
            data=df_implausibility, x="Implausibility", hue="Wave", palette=self.n_waves*["#000000"],
            cumulative=True, common_norm=False, common_grid=True, alpha=1, lw=2, legend=False
        )
        ax.axvline(self.threshold_final, color='#bc6c25', linestyle="--", linewidth=2)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.tick_params(width=1.5)
        ax.set(xlim=(0, 10), ylim=(0, 1), xlabel="Emulated implausibility (-)", ylabel="Density (-)")
        ax.set_box_aspect(1)

        # Save figure
        plt.tight_layout()
        plt.savefig(filepath / filename, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_implausibility_counts(self, filepath=None, filename="implausibility_most.pdf"):
        """Stacked bar chart containing y with the highest implausibility for each wave"""

        if not filepath:
            filepath = self.dir_output

        # Collect fractions of highest implausibility for each y in all waves
        fractions = np.zeros((self.n_waves, self.n_y))

        for i_wave, wave in enumerate(self.waves):

            # Identify which y had the highest implausibility
            i_max = np.argmax(wave.implausibility_y, axis=1)

            # Get fraction of max implausibility of each y
            for i_y in range(self.n_y):
                fractions[i_wave, i_y] = np.count_nonzero(i_max == i_y)/wave.n_emu

        # Convert to Pandas dataframe
        df_fractions = pd.DataFrame(data=fractions, columns=self.y_names, index=self.wave_numbers)

        # Plot results
        fig, ax = plt.subplots(figsize=(4.5, self.n_waves))
        df_fractions.plot(ax=ax, kind='bar', stacked=True, edgecolor='k', width=1.0, color=self.cmap_y)

        ax.set(ylim=(0, 1), xlim=(-0.5, self.n_waves-0.5), yticks=(0.0, 0.5, 1.0), xlabel="Wave", ylabel="Most implausible output (-)")
        ax.set_xticklabels(self.wave_numbers, rotation=0)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='center left', bbox_to_anchor=(1, 0.5), fontsize="small", frameon=False)

        ax.set_box_aspect(1.25)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.tick_params(width=1.5)

        plt.tight_layout()
        plt.savefig(filepath / filename, bbox_inches='tight', dpi=300)
        plt.close()

    def get_sensitivity_matrix(self, order="ST", i_wave=-1):
        """Collect sensitivity matrices of all waves"""

        # Collect sensitivity matrix
        if self.waves[i_wave].sobol_indices is None:
            print("No sensitivity analysis performed during wave " + self.waves[-1].name)
            sensitivity_matrix = None
        else:
            sensitivity_matrix = np.zeros((self.n_x, self.n_y))
            for i_y in range(self.n_y):
                sensitivity_matrix[:, i_y] = np.maximum(self.waves[i_wave].sobol_indices[i_y][order][:self.n_x], 0)

        return sensitivity_matrix

    def plot_sensitivity_matrix(self, filepath=None, filename="gsa.pdf", order="ST", show_fig=False,
                                x_fontsize=10, y_fontsize=10, tile_size=0.25):
        """Plot results of Sobol sensitivity analysis"""

        if not filepath:
            filepath = self.dir_output

        # Plot sensitivity if an analysis was performed during a wave
        for i_wave in range(self.n_waves):
            wave = self.waves[i_wave]
            if wave.sobol_indices is not None:

                # Get sensitivity matrix
                sense_matrix = self.get_sensitivity_matrix(order=order, i_wave=i_wave)

                # Part 2a: Plot sensitivity matrix of first-order effects
                df = pd.DataFrame(data=sense_matrix, columns=self.y_names, index=self.x_names[:self.n_x])

                # fig, ax = plt.subplots()
                fig, ax = plt.subplots(figsize=(tile_size * self.n_x, tile_size * self.n_y))
                ax.set_title("Global Sensitivity Analysis (" + order + ")")

                ax_divider = make_axes_locatable(ax)
                cax1 = ax_divider.append_axes("right", size="4%", pad="6%")
                ax = sns.heatmap(df, ax=ax, linewidth=1, cmap=sns.cubehelix_palette(as_cmap=True), square=True,
                                 cbar_ax=cax1, cbar_kws={"ticks": [0.0, 0.5, 1.0]})
                ax.tick_params(left=False, bottom=False)
                ax.set_xlabel("Outputs", fontsize=x_fontsize)
                ax.set_ylabel("Inputs", fontsize=y_fontsize)
                ax.tick_params(axis='x', rotation=45)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                plt.tight_layout()
                plt.savefig(filepath / str(filename.split(".")[0] + "_matrix_" + "order" + "_" + wave.name + "." +
                                           filename.split(".")[1]), bbox_inches='tight')
                if show_fig:
                    plt.show()
                plt.close()

    def plot_sensitivity_total(self, filepath=None, filename="gsa.pdf", show_fig=False, cutoff=0.90, fontsize=8,
                               order="ST"):
        """Plot results of Sobol sensitivity analysis, total sensitivity only"""

        if not filepath:
            filepath = self.dir_output

        # Plot sensitivity if an analysis was performed during a wave
        for i_wave in range(self.n_waves):
            wave = self.waves[i_wave]
            if wave.sobol_indices is not None:

                # Get sensitivity matrix
                sense_matrix = self.get_sensitivity_matrix(order=order, i_wave=i_wave)

                # Get total effects
                total_weight = np.sum(sense_matrix, axis=1) / np.sum(sense_matrix)
                df = pd.DataFrame(data=[total_weight],
                                  columns=list(self.x_names[:self.n_x])).sort_values(by=0, axis=1, ascending=False)

                # Only keep parameters of which the sum of all is above the cutoff
                for i in range(len(df.columns)):
                    sum_sobol = np.sum(df.iloc[0, :i])
                    if sum_sobol > cutoff:
                        df = df.iloc[:, :i]
                        # Add column with remaining sensitivity
                        df["Remaining"] = 1 - sum_sobol
                        break

                cmap = sns.color_palette("cubehelix", len(df.columns) - 1, as_cmap=False)
                # Add white color for remaining sensitivity
                cmap.append("#ffffff")

                fig, ax = plt.subplots(figsize=(5, 2))
                for x in np.linspace(0, 1, 5):
                    ax.plot([x] * 2, [-.58, -0.25], ":", zorder=1, color="#000000", linewidth=2)
                    ax.text(x, -0.9, str(int(x * 100)) + "%", horizontalalignment="center", color="#000000")
                df.plot(ax=ax, kind="barh", stacked=True, legend=False, color=cmap, edgecolor="black",
                        linewidth=1.5, zorder=2)
                ax.axis('off')
                ax.set(ylim=(-1, 1), xlim=(-0.1, 1.15))
                fig.suptitle("Total parameter sensitivity")

                # Add input labels to plot
                for p, x_name in zip(ax.patches, list(df.columns)):
                    width, height = p.get_width(), p.get_height()
                    x, y = p.get_xy()
                    ax.text(x + width / 2, y + height * 1.2, x_name, fontsize=fontsize,
                            horizontalalignment='left', verticalalignment='bottom', rotation=45)

                plt.tight_layout()
                plt.savefig(filepath / str(filename.split(".")[0] + "_total_" + wave.name + "." +
                                           filename.split(".")[1]), bbox_inches='tight')
                if show_fig:
                    plt.show()
                plt.close()

                # Make circle plot of total sensitivity
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.set_title("Total parameter sensitivity")
                ax.pie(df.iloc[0, :], labels=df.columns, colors=cmap, startangle=0,
                       wedgeprops={'edgecolor': 'white', 'width': 0.3, 'linewidth': 3},
                       textprops={'color': '#525252', 'fontsize': 10})

                # wedges, texts = ax.pie(np.append(s_12, s_remaining), colors=cmap, startangle=0,
                #                        wedgeprops={'width': 0.4, 'linewidth': 4, 'edgecolor': 'white'},
                #                        textprops={'color': '#525252'})

                plt.tight_layout()
                plt.savefig(filepath / str(filename.split(".")[0] + "_total_pie_" + wave.name + "." +
                                             filename.split(".")[1]), bbox_inches='tight')
                if show_fig:
                    plt.show()
                plt.close()

    def plot_sensitivity_circles(self, filepath=None, filename="gsa.pdf", show_fig=False, cutoff=0.90):
        """Plot results of Sobol sensitivity analysis"""

        if not filepath:
            filepath = self.dir_output

        # Plot sensitivity if an analysis was performed during a wave
        for wave in self.waves:
            if wave.sobol_indices is not None:

                # Part 1: Start pie chart figure
                fig = plt.figure(figsize=(2.5*wave.plot_shape_y[1], 2.5*wave.plot_shape_y[0]))
                plt.suptitle("Parameter sensitivity")

                # Add higher-order terms color and legend entry (off-white)
                cmap = self.cmap_x.copy()
                cmap.append("#f5f6fa")
                x_names = self.x_names + ['higher-order']
                wedges = []

                # Add first, second or total order effects to sensitivity matrix
                for i_y in range(self.n_y):
                    sobol_indices = np.maximum(wave.sobol_indices[i_y]['ST'][:self.n_x], 0)

                    # Get total effects
                    df = pd.DataFrame(data=[sobol_indices],
                                      columns=list(self.x_names[:self.n_x])).sort_values(by=0, axis=1, ascending=False)

                    # Only keep parameters of which the sum of all is above the cutoff
                    for i in range(len(df.columns)):
                        sum_sobol = np.sum(df.iloc[0, :i])
                        if sum_sobol > cutoff:
                            df = df.iloc[:, :i]
                            # Add column with remaining sensitivity
                            df[""] = 1 - sum_sobol
                            break

                    cmap = sns.color_palette("cubehelix", len(df.columns) - 1, as_cmap=False)
                    # Add white color for remaining sensitivity
                    cmap.append("#ffffff")

                    ax = fig.add_subplot(wave.plot_shape_y[0], wave.plot_shape_y[1], i_y+1)
                    ax.set_title(self.y_names[i_y])

                    wedges, texts = ax.pie(df.iloc[0, :], labels=df.columns, colors=cmap, startangle=0,
                                           wedgeprops={'width': 0.4, 'linewidth': 3, 'edgecolor': 'white'},
                                           textprops={'color': '#525252', 'fontsize': 8})

                # fig.legend(wedges, x_names, loc='center left', bbox_to_anchor=(1, 0.5))
                plt.tight_layout()
                plt.savefig(filepath / str(filename.split(".")[0] + "_pies_" + wave.name + "." +
                                           filename.split(".")[1]), bbox_inches='tight')
                plt.tight_layout()
                if show_fig:
                    plt.show()
                plt.close()

    def plot_connectivity(self, cutoff=0.05, filepath=None, filename="connectivity.pdf", show_fig=False,
                          cmap="cubehelix"):
        """Plot connectivity of Sobol sensitivity analysis"""

        if not filepath:
            filepath = self.dir_output

        for wave in self.waves:
            if wave.sobol_problem is not None:

                # Collect total order effects from inputs on outputs, and secondary interactions between inputs
                connectivity = np.zeros((self.n_x+self.n_y, self.n_x+self.n_y))
                for i_y in range(self.n_y):

                    # Obtain total order effects, if any. Absence of secondary-effects is indicated by NaNs so ignore those
                    s_total = np.maximum(wave.sobol_indices[i_y]['ST'][:self.n_x], 0)

                    # Add total interaction between inputs and output
                    connectivity[self.n_x + i_y, :self.n_x] = s_total

                    # # Add interactions effects between parameters to matrix
                    # for i1 in range(self.n_x):
                    #     for i2 in range(self.n_x):
                    #         connectivity[i2, i1] += wave.sp.analysis[self.y_names[i_y]]['S2'][i1, i2]

                # Don't display weak links
                connectivity = np.where(connectivity > cutoff, connectivity, np.nan)

                # Node width
                node_width = 360 / len(self.x_names + self.y_names)

                # Node angles, start with input nodes around 180 degrees, then fill with output nodes
                node_angles = (180 - len(self.x_names) / 2 * node_width +
                               0.5 * node_width + np.arange(self.n_x) * node_width)
                node_angles = np.append(node_angles, node_angles[-1] + np.arange(1, self.n_y + 1) * node_width)

                # Node colors
                node_colors = [(0.7, 0.7, 0.7)] * self.n_x + sns.color_palette(cmap, self.n_y, as_cmap=False)

                # Plot black background
                fig, ax = plt.subplots(figsize=(8, 8), facecolor='white',
                                       subplot_kw=dict(polar=True))
                plot_connectivity_circle(connectivity, self.x_names[:self.n_x] + self.y_names, ax=ax, node_angles=node_angles,
                                         title="Model connectivity", show=False, node_colors=node_colors, linewidth=5,
                                         colorbar=True, facecolor='white', textcolor='black', colormap="Greys")
                fig.tight_layout()
                fig.savefig(filepath / str(filename.split(".")[0] + "_" + wave.name + "." +
                                           filename.split(".")[1]), bbox_inches='tight')
                if show_fig:
                    plt.show()
                plt.close()

    def plot_nroy_x(self, filepath=None, filename="nroy_x.pdf", wave_id=-1, n_samples=None, n_levels=10,
                    show_fig=False):
        """Plot NROY region distribution of parameters at a specific wave, default final wave"""

        if not filepath:
            filepath = self.dir_output

        # Obtain wave
        wave = self.waves[wave_id]

        # Plot variance space for each y variable
        df = pd.DataFrame(data=wave.nroy, columns=self.x_names)

        # Sample data if desired
        if n_samples is not None:
            if n_samples < len(df):
                df = df.sample(n_samples)

        # Plot NROY region evolution over the waves
        g = sns.PairGrid(df, vars=self.x_names_matched, corner=True, height=1.25, aspect=1,
                         despine=False, diag_sharey=False)
        g.map_lower(sns.kdeplot, cmap=sns.cubehelix_palette(as_cmap=True), fill=True, levels=n_levels)
        g.map_diag(sns.kdeplot, fill=True, color=self.cmap_waves[-1])

        # Draw vertical lin in diagonal plots
        if self.x_target is not None:
            for i in range(len(self.x_names_matched)):
                g.axes[i, i].axvline(self.x_target[i], color=[0.5, 0.5, 0.5], linestyle='-', linewidth=1)

        g.fig.suptitle('Parameter space in NROY region after ' + wave.name)
        plt.savefig(filepath / str(filename.split(".")[0] + "_" + wave.name + "." + filename.split(".")[1]), bbox_inches='tight')

        if show_fig:
            plt.show()

        plt.close()

    def plot_nroy_y(self, filepath=None, filename="nroy_y.pdf", wave_id=-1, n_samples=None, n_levels=10, lims=None,
                    show_fig=False):
        """Plot NROY region distribution of outputs at a specific wave, default final wave. Lims can be set to a
        numerical value that is lims*sigma_observed away from the observed value"""

        if not filepath:
            filepath = self.dir_output

        # Obtain wave
        wave = self.waves[wave_id]

        # Plot variance space for each y variable
        df = pd.DataFrame(data=wave.nroy_y, columns=self.y_names)

        # Sample data if desired
        if n_samples is not None:
            if n_samples < len(df):
                df = df.sample(n_samples)

        # Plot NROY region evolution over the waves
        g = sns.PairGrid(df, vars=self.y_names, corner=True, height=1.25, aspect=1,
                         despine=False, diag_sharey=False)
        g.map_lower(sns.kdeplot, cmap=sns.cubehelix_palette(as_cmap=True), fill=True, levels=n_levels)
        g.map_diag(sns.kdeplot, fill=True, color=self.cmap_waves[-1])

        # Draw vertical lin in diagonal plots
        for i in range(len(self.y_names)):
            g.axes[i, i].axvline(self.y_observed[i], color=[0.5, 0.5, 0.5], linestyle='-', linewidth=1)
            g.axes[i, i].axvline(self.y_observed[i] - self.confidence_lim * self.sigma_observed[i], color=[0.5, 0.5, 0.5], linestyle=':', linewidth=2)
            g.axes[i, i].axvline(self.y_observed[i] + self.confidence_lim * self.sigma_observed[i], color=[0.5, 0.5, 0.5], linestyle=':', linewidth=2)

        # Set axis limits
        for i in range(len(self.y_names)):
            g.axes[i, i].set_xlim([self.y_observed[i] - 3 * self.sigma_observed[i], self.y_observed[i] + 3 * self.sigma_observed[i]])

        # Set y limits for off diagional plots
        if lims is not None:
            for i in range(len(self.y_names)):
                for j in range(i):
                    g.axes[i, j].set_ylim([self.y_observed[i] - 3 * self.sigma_observed[i], self.y_observed[i] + 3 * self.sigma_observed[i]])

        g.fig.suptitle('Output space in NROY region after ' + wave.name)
        plt.savefig(filepath / str(filename.split(".")[0] + "_" + wave.name + "." + filename.split(".")[1]), bbox_inches='tight')

        if show_fig:
            plt.show()

        plt.close()

    def plot_nroy_waves(self, filepath=None, filename="nroy_waves.png", show_fig=False, sample_fraction=1.0,
                        tile_size=1.25):
        """Plot the multidimensional NROY space of all waves, including the initial parameters space"""

        if not filepath:
            filepath = self.dir_output

        # Add initial parameter space
        df_nroy = pd.DataFrame(data=self.waves[0].x_emu, columns=self.x_names)
        df_nroy["Wave"] = "Initial"

        for wave in self.waves:
            df = pd.DataFrame(data=wave.nroy, columns=self.x_names)
            df["Wave"] = wave.name

            # Sample data if desired
            if sample_fraction is not None:
                n_samples = int(sample_fraction * len(df))
                if n_samples < len(df):
                    df = df.sample(n_samples)

            # Add to compiled spreadsheet
            df_nroy = pd.concat([df_nroy, df])

        # Plot NROY region evolution over the waves
        g = sns.PairGrid(df_nroy, vars=self.x_names, hue="Wave", corner=True, height=tile_size, aspect=1,
                         despine=False, palette=self.cmap_waves)
        g.map(sns.scatterplot, edgecolor=None, s=5)        # Used to be s=15
        g.add_legend(fontsize="small", title="")
        g._legend.set_bbox_to_anchor((0.65, 0.77))
        g.savefig(filepath / filename, bbox_inches='tight', dpi=300)

        if show_fig:
            plt.show()

        plt.close()

    def plot_sim_space(self, filepath=None, filename="sim_space.png", show_fig=False):
        """Plot the multidimensional NROY space of all waves, including the initial parameters space"""

        if not filepath:
            filepath = self.dir_output

        # Start with first wave...
        df_sims = pd.DataFrame(data=self.waves[-1].x_sim, columns=self.x_names)

        # Plot NROY region evolution over the waves
        g = sns.PairGrid(df_sims, vars=self.x_names_matched, corner=True, height=1.25, aspect=1,
                         despine=False)
        g.map(sns.scatterplot, edgecolor=None, s=15, alpha=0.1, color=self.cmap_waves[int(self.n_waves/2)])
        g.add_legend(fontsize="small", title="")

        g.fig.suptitle("Simulation data cloud")
        g.savefig(filepath / filename, bbox_inches='tight', dpi=300)

        if show_fig:
            plt.show()

        plt.close()

    def plot_emulation(self, i_wave=-1, filepath=None, filename="emulation.png", show_fig=False):
        """Plot multivariate emulation"""

        if not filepath:
            filepath = self.dir_output

        fig = plt.figure(figsize=(2.5*self.waves[i_wave].plot_shape_y[1], 2.5*self.waves[i_wave].plot_shape_y[0]))

        # Normalize emulation inputs and variance
        x_emu = scale(self.waves[i_wave].x_emu, self.waves[-i_wave].x_scaler)
        y_emu = self.waves[i_wave].y_emu
        y_emu_variance = self.waves[i_wave].y_emu_variance

        # Normalize simulation inputs
        x_sim = scale(self.waves[i_wave].x_sim, self.waves[-i_wave].x_scaler)
        y_sim = self.waves[i_wave].y_sim

        # Plot emulations and confidence intervals
        for i_y in range(self.n_y):
            ax = fig.add_subplot(self.waves[i_wave].plot_shape_y[0], self.waves[i_wave].plot_shape_y[1], i_y+1)

            ax.set_title(self.y_names[i_y])

            # Data points, sort on ascending validation y
            for i_x in range(self.n_x):
                i_sort = np.argsort(x_emu[:, i_x])
                ax.plot(x_emu[i_sort, i_x], y_emu[i_sort, i_y], color=self.cmap_x[i_x], zorder=10, linewidth=3)

                ax.scatter(x_sim[:, i_x], y_sim[:, i_y], color='k', s=2, zorder=15, edgecolor='k')

                ax.fill_between(x_emu[i_sort, i_x],
                                y_emu[i_sort, i_y] - np.sqrt(y_emu_variance[i_sort, i_y])*2,
                                y_emu[i_sort, i_y] + np.sqrt(y_emu_variance[i_sort, i_y])*2,
                                color=self.cmap_x[i_x], alpha=0.5, zorder=0)

            ax.set(xlim=(0, 1.0), xticks=(0, 0.5, 1.0))
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.set_box_aspect(1)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2)

        plt.tight_layout()
        plt.savefig(filepath / filename, bbox_inches='tight')

        if show_fig:
            plt.show()

        plt.close()

    def plot_nroy_data(self, i_wave=-1, filepath=None, show_box=False, show_violin=False, filename="emu_data.pdf"):

        if not filepath:
            filepath = self.dir_output

        # Emulation box plots, normalize and plot data
        nroy_y_emu = normalize_data_ci(self.waves[i_wave].nroy_y, self.waves[i_wave].y_observed,
                                       self.waves[i_wave].sigma_observed)
        df_nroy_emu = pd.DataFrame(data=nroy_y_emu, columns=self.y_names)
        plot_boxes_violins(df_nroy_emu, filepath=filepath, filename=filename,
                           show_violin=show_violin, show_box=show_box)

        # Posterior box plots, normalize and plot data
        if self.waves[i_wave].y_posterior is not None:
            nroy_y_posterior = normalize_data_ci(self.waves[i_wave].y_posterior, self.waves[i_wave].y_observed,
                                                 self.waves[i_wave].sigma_observed)
            df_nroy_posterior = pd.DataFrame(data=nroy_y_posterior, columns=self.y_names)
            plot_boxes_violins(df_nroy_posterior, filepath=filepath, filename="posterior_data.pdf")


def plot_boxes_violins(df_nroy, filepath=None, filename="nroy_data.pdf", show_box=False, show_violin=False, cmap=None):
    """Plot box and violin plots of outputs normalized and compared to 95% data confidence interval"""

    # Amount of data points equals number of columns of df_nroy
    n_y = df_nroy.shape[1]

    # Use cubehelix cmap if not specified
    if cmap is None:
        cmap = sns.color_palette("cubehelix", n_y, as_cmap=False)

    # Use current directory if not specified
    if filepath is None:
        filepath = pathlib.Path().absolute()

    filenames = [filename.split(".")[0] + "_violin." + filename.split(".")[-1],
                 filename.split(".")[0] + "_box." + filename.split(".")[-1]]

    for i in range(2):

        if i == 0:
            fig, ax = plt.subplots(figsize=(1*n_y, 5))
            sns.violinplot(data=df_nroy, palette=cmap, bw_adjust=1.0, linewidth=1.5, zorder=5)
        else:
            fig, ax = plt.subplots(figsize=(0.75*n_y, 5))
            sns.boxplot(data=df_nroy, palette=cmap, linewidth=1.5, zorder=5, showfliers=False)

        # Draw horizontal line at 0
        plt.axhline(y=0, color='k', linestyle='-', linewidth=1.5, alpha=0.3, zorder=0)
        plt.axhline(y=-0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.3, zorder=0)
        plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.3, zorder=0)
        plt.fill_between([-0.5, n_y-0.5], -0.5, 0.5, color='k', alpha=0.1, zorder=0, linewidth=0)

        ax.set(xlim=(-0.5, n_y-0.5), ylim=(-1.0, 1.0 ), ylabel="Normalized outputs",
               yticks=(-0.5, 0, 0.5), yticklabels=("-2std", "mean", "+2std"))
        ax.tick_params(left=False, bottom=False)
        ax.tick_params(axis='x', rotation=45)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

        plt.savefig(filepath / filenames[i], bbox_inches='tight')

        if (i == 0) and show_violin:
            plt.show()
        elif (i == 1) and show_box:
            plt.show()

        plt.close()
