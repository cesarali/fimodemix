
from torch import nn

class StateDependentSDEFigures(Task):
    def __init__(self, **kwargs) -> None:
        self.model: nn.Module = kwargs["model_dict"].get("model")
        self.num_devices = len(jax.local_devices())

        model_call_method = "vmap_model_call" if self.num_devices == 1 else "pmap_model_call"
        self.model_call: Callable = kwargs["model_dict"].get(model_call_method)

        data_generator: str = kwargs.get("data_generator")
        self.data_generator: DataGenerator = kwargs["dataloader"].get_data_generator(data_generator)

        self.plot_observations = kwargs.get("plot_observations", True)
        self.plot_paths_count: int = kwargs.get("plot_paths_count")

        self.em_steps: int = kwargs.get("em_steps", 8)

        self._2D_subsample_locations_by_stride: int = kwargs.get("2D_subsample_locations_by_stride", 1)

        self._1D_location_count: int = kwargs.get("1D_location_count", -1)
        self._2D_location_count: int = kwargs.get("2D_location_count", -1)
        self._3D_location_count: int = kwargs.get("3D_location_count", -1)

        self.enable_1D_figure: bool = kwargs.get("enable_1D_figure", True)
        self.enable_2D_figure: bool = kwargs.get("enable_2D_figure", True)
        self.enable_3D_figure: bool = kwargs.get("enable_3D_figure", True)

        self.plot_std_bands: bool = kwargs.get("plot_std_bands", True)

        self.paths_linewidth: float = kwargs.get("paths_linewidth", 1.0)

    def run(self, state: TrainState):
        "Return figures in dict (based on test set) to monitore in tensorboard."
        generator_key, apply_key, index_key = jax.random.split(state.key, 3)

        minibatch = next(self.data_generator.get_generator(generator_key, state.epoch))

        call_input = self.model.apply(
            state.variables, minibatch, method=self.model.get_call_input_from_minibatch, rngs={"sample": PRNGKey(0)}
        )
        call_input = (False,) + tuple(call_input)
        model_outputs, _ = self.model_call(state.variables, apply_key, *call_input)

        if self.num_devices > 1:
            # drop device axis
            minibatch, *model_outputs = jax.tree_util.tree_map(
                lambda x: x.reshape((-1,) + x.shape[2:]) if x.ndim > 2 else x.reshape(-1, 1),
                (minibatch, *model_outputs),
            )

        # select random time series to present
        index = jax.random.choice(index_key, a=minibatch["obs_values"].shape[0])
        minibatch, *model_outputs = jax.tree_util.tree_map(lambda x: x[index], (minibatch, *model_outputs))
        loss_input = self.model.apply(
            state.variables, minibatch, method=self.model.get_loss_input_from_minibatch, rngs={"sample": PRNGKey(0)}
        )

        # figure type depends on equation dimension of selected time series
        multi_dim_mask = loss_input[9]
        if multi_dim_mask.sum().item() == 3.0 and self.enable_1D_figure:
            return self._3D_figure(state, loss_input, model_outputs)
        elif multi_dim_mask.sum().item() == 2.0 and self.enable_2D_figure:
            return self._2D_figure(state, loss_input, model_outputs)
        elif multi_dim_mask.sum().item() == 1.0 and self.enable_3D_figure:
            return self._1D_figure(state, loss_input, model_outputs)
        else:
            return state, {}

    def _1D_figure(self, state: TrainState, loss_input: tuple, model_outputs: tuple) -> tuple:
        (
            obs_times,
            obs_values,
            obs_mask,
            fine_delta_x,
            locations_on_hypercube,
            target_concepts_on_hypercube,
            locations_on_path,
            target_concepts_on_paths,
            paths_mask,
            multi_dim_mask,
            ground_truth_init_cond_distr_params,
            drift_loss_scale,
            diffusion_loss_scale,
            init_cond_loss_scale,
            loss_threshold,
        ) = loss_input

        (
            concept_dist_params_at_locations,
            init_cond_dist_params,
            normalized_concept_dist_params_at_locations,
            normalized_init_cond_dist_params,
            obs_values_norm_params,
            obs_times_norm_params,
        ) = model_outputs

        # need to truncate location list to only contain the original input lenghts before padding during dataloading
        locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations = jax.tree_util.tree_map(
            lambda x: x[..., : self._1D_location_count, :],
            (locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations),
        )

        # need to truncate observations, locations and concepts to 1D, their original dimensions before padding
        locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations, obs_values = jax.tree_util.tree_map(
            lambda x: x[..., :1], (locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations, obs_values)
        )

        if obs_mask is None:
            obs_mask = jnp.ones_like(obs_times)

        # sde: f(x) dt + g(x) dW
        if len(concept_dist_params_at_locations.drift_terms_mean) == 1:
            drift_mean = concept_dist_params_at_locations.drift_terms_mean[0]
            drift_log_std = concept_dist_params_at_locations.drift_terms_log_std[0]
            diffusion_mean = concept_dist_params_at_locations.diffusion_terms_mean[0]
            diffusion_log_std = concept_dist_params_at_locations.diffusion_terms_log_std[0]

            target_drift = target_concepts_on_hypercube.drift_terms[0]
            target_diffusion = target_concepts_on_hypercube.diffusion_terms[0]

        else:
            raise ValueError(
                "This method can only plot one function per drift and diffusion, got"
                f" {len(concept_dist_params_at_locations.drift_terms_mean)}."
            )

        # locations might be unordered, sort it
        sort_indices = jnp.argsort(locations_on_hypercube, axis=0)
        locations_on_hypercube, drift_mean, drift_log_std, diffusion_mean, diffusion_log_std, target_drift, target_diffusion = (
            jax.tree_util.tree_map(
                lambda x: x[sort_indices],
                (locations_on_hypercube, drift_mean, drift_log_std, diffusion_mean, diffusion_log_std, target_drift, target_diffusion),
            )
        )

        # plot drift and diffusion on different axes of the same figure
        # plot drift and diffusion value (y-axis) depending on the location (x-axis)
        fig, ax1 = plt.subplots(dpi=300)
        ax1_color = "tab:red"
        ax1.tick_params(axis="y", labelcolor=ax1_color)
        ax1.set_ylabel("Drift", color=ax1_color)
        ax1.set_xlabel("State", color="black")

        ax2 = ax1.twinx()
        ax2_color = "tab:blue"
        ax2.tick_params(axis="y", labelcolor=ax2_color)
        ax2.set_ylabel("Diffusion", color=ax2_color)

        # plot ground truth concepts
        ax1.plot(
            locations_on_hypercube.reshape(-1),
            target_drift.reshape(-1),
            label="Ground-Truth Drift",
            c=ax1_color,
        )
        ax2.plot(
            locations_on_hypercube.reshape(-1),
            target_diffusion.reshape(-1),
            label="Ground-Truth Diffusion",
            c=ax2_color,
            linestyle="dashed",
        )

        # plot inferred concepts
        ax1.plot(
            locations_on_hypercube.reshape(-1),
            drift_mean.reshape(-1),
            c="black",
            label="Mean Drift",
        )
        ax2.plot(
            locations_on_hypercube.reshape(-1),
            diffusion_mean.reshape(-1),
            c="black",
            linestyle="dashed",
            label="Mean Diffusion",
        )

        if self.plot_std_bands is True:
            # plot inferred std bands
            ax1.fill_between(
                locations_on_hypercube.reshape(-1),
                (drift_mean + jnp.exp(drift_log_std)).reshape(-1),
                (drift_mean - jnp.exp(drift_log_std)).reshape(-1),
                color=ax1_color,
                alpha=0.35,
                zorder=0,
            )
            ax2.fill_between(
                locations_on_hypercube.reshape(-1),
                (diffusion_mean + jnp.exp(diffusion_log_std)).reshape(-1),
                (diffusion_mean - jnp.exp(diffusion_log_std)).reshape(-1),
                color=ax2_color,
                alpha=0.35,
                zorder=0,
            )

        fig.legend(ncol=4)

        fig_dict = {"1D_vector_field": fig}

        # plot data paths to check and samples
        if self.plot_observations is True:
            fig_paths, ax_paths = plt.subplots(dpi=300)
            fig_paths.suptitle("Observation Paths")
            ax_paths.tick_params(axis="y")
            ax_paths.set_ylabel("Path")

            obs_config = {"c": ax1_color, "linewidth": self.paths_linewidth}
            samples_config = {"c": "black", "linewidth": self.paths_linewidth}

            plot_1D_observations_and_samples(
                self,
                state,
                ax_paths,
                obs_times,
                obs_values,
                obs_mask,
                paths_mask,
                locations_on_hypercube,
                fine_delta_x,
                # normalized_init_cond_dist_params,
                init_cond_dist_params,
                obs_config,
                samples_config,
            )

            fig_paths.legend(ncol=4)
            fig_dict.update({"1D_Paths": fig_paths})

            plt.close(fig_paths)

        plt.close()
        return state, fig_dict

    def _2D_figure(self, state: TrainState, loss_input: tuple, model_outputs: tuple) -> tuple:
        (
            obs_times,
            obs_values,
            obs_mask,
            fine_delta_x,
            locations_on_hypercube,
            target_concepts_on_hypercube,
            locations_on_path,
            target_concepts_on_paths,
            paths_mask,
            multi_dim_mask,
            ground_truth_init_cond_distr_params,
            drift_loss_scale,
            diffusion_loss_scale,
            init_cond_loss_scale,
            loss_threshold,
        ) = loss_input

        (
            concept_dist_params_at_locations,
            init_cond_dist_params,
            normalized_concept_dist_params_at_locations,
            normalized_init_cond_dist_params,
            obs_values_norm_params,
            obs_times_norm_params,
        ) = model_outputs

        # need to truncate location list to only contain the original input lenghts before padding during dataloading
        locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations = jax.tree_util.tree_map(
            lambda x: x[..., : self._2D_location_count, :],
            (locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations),
        )

        # need to truncate observations, locations and concepts to 2D, their original dimensions before padding
        locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations, obs_values = jax.tree_util.tree_map(
            lambda x: x[..., :2], (locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations, obs_values)
        )

        if obs_mask is None:
            obs_mask = jnp.ones_like(obs_times)

        fig, axs = plt.subplots(2, 2, figsize=(7, 7), tight_layout=True, dpi=300)

        def _plot_vector_field(ax, locs, vals, scale=None):
            """
            Plot 2D vector field as quivers.
            ax: axis to plot into
            loc: locations [L, 2], vals: [L, 2]
            scale: scale of quivers from another plot, to use the same scale in this plot
            """

            # reshape locations into a 2D grid
            point_per_dim = int(math.sqrt(locs.shape[0]))
            locs = locs[: point_per_dim**2].reshape(point_per_dim, point_per_dim, 2)
            vals = vals[: point_per_dim**2].reshape(point_per_dim, point_per_dim, 2)

            # optionally subsample all locations for cleaner plots with fewer quivers
            locs = locs[:: self._2D_subsample_locations_by_stride, :: self._2D_subsample_locations_by_stride]
            vals = vals[:: self._2D_subsample_locations_by_stride, :: self._2D_subsample_locations_by_stride]

            # reshape again into lists for quiver plotting
            locs = locs.reshape(-1, 2)
            vals = vals.reshape(-1, 2)

            # split dimensions for plotting in ax.quiver
            x_locations, y_locations = jnp.split(locs, indices_or_sections=2, axis=-1)
            x_vals, y_vals = jnp.split(vals, indices_or_sections=2, axis=-1)

            # scale quivers of scale of other quivers, if available
            quiver = ax.quiver(x_locations, y_locations, x_vals, y_vals, scale=scale)

            return quiver

        # scale quivers of inferred concepts based on ground truth quiver lengths
        drift_quiver = _plot_vector_field(axs[0, 0], locations_on_hypercube, target_concepts_on_hypercube.drift_terms[0])
        drift_quiver._init()

        diffusion_quiver = _plot_vector_field(axs[0, 1], locations_on_hypercube, target_concepts_on_hypercube.diffusion_terms[0])
        diffusion_quiver._init()

        # plot inferred concepts
        _plot_vector_field(
            axs[1, 0], locations_on_hypercube, concept_dist_params_at_locations.drift_terms_mean[0], scale=drift_quiver.scale
        )
        _plot_vector_field(
            axs[1, 1], locations_on_hypercube, concept_dist_params_at_locations.diffusion_terms_mean[0], scale=diffusion_quiver.scale
        )

        axs[0, 0].set_title("Drift")
        axs[0, 1].set_title("Diffusion")

        axs[0, 0].set_ylabel("Ground Truth")
        axs[1, 0].set_ylabel("Model")

        # fig.legend(ncol=4)
        fig_dict = {"2D_vector_field": fig}
        plt.close()

        # plot data paths to check and samples
        if self.plot_observations is True:
            fig_paths, ax_paths = plt.subplots(dpi=300)
            fig_paths.suptitle("Observation Paths")
            ax_paths.tick_params(axis="y")
            ax_paths.set_ylabel("Path")

            obs_config = {"c": "red", "linewidth": self.paths_linewidth}
            samples_config = {"c": "black", "linewidth": self.paths_linewidth}
            init_state_config = {"color": "blue", "s": 3}

            plot_2D_observations_and_samples(
                self,
                state,
                ax_paths,
                obs_times,
                obs_values,
                obs_mask,
                paths_mask,
                locations_on_hypercube,
                fine_delta_x,
                # normalized_init_cond_dist_params,
                init_cond_dist_params,
                obs_config,
                samples_config,
                init_state_config,
            )

            fig_paths.legend(ncol=4)
            fig_dict.update({"2D_Paths": fig_paths})

        return state, fig_dict

    def _3D_figure(self, state: TrainState, loss_input: tuple, model_outputs: tuple) -> tuple:
        (
            obs_times,
            obs_values,
            obs_mask,
            fine_delta_x,
            locations_on_hypercube,
            target_concepts_on_hypercube,
            locations_on_path,
            target_concepts_on_paths,
            paths_mask,
            multi_dim_mask,
            ground_truth_init_cond_distr_params,
            drift_loss_scale,
            diffusion_loss_scale,
            init_cond_loss_scale,
            loss_threshold,
        ) = loss_input

        (
            concept_dist_params_at_locations,
            init_cond_dist_params,
            normalized_concept_dist_params_at_locations,
            normalized_init_cond_dist_params,
            obs_values_norm_params,
            obs_times_norm_params,
        ) = model_outputs

        # need to truncate location list to only contain the original input lenghts before padding during dataloading
        locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations = jax.tree_util.tree_map(
            lambda x: x[..., : self._3D_location_count, :],
            (locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations),
        )

        # need to truncate observations, locations_on_hypercube and concepts to 2D, their original dimensions before padding
        locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations = jax.tree_util.tree_map(
            lambda x: x[..., :3], (locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations)
        )

        if obs_mask is None:
            obs_mask = jnp.ones_like(obs_times)

        # plot a grid with 4 rows: ground-truth drift, model drift, ground-truth diffusion, model diffusion
        # select a slice in the first dimension
        # plot concepts in this slice per dimension
        fig, axs = plt.subplots(4, 3, figsize=(7, 7), tight_layout=True, dpi=300)

        # reshape concepts into 3D grid
        points_per_dim = math.floor(locations_on_hypercube.shape[0] ** (1 / 3))
        locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations = jax.tree_util.tree_map(
            lambda x: x[: points_per_dim**3].reshape(points_per_dim, points_per_dim, points_per_dim, 3),
            (locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations),
        )

        # sort by locations, for more coherent image, should they not be ordered
        for axis in range(3):
            sort_indices = jnp.argsort(locations_on_hypercube, axis=axis)
            locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations = jax.tree_util.tree_map(
                lambda x: jnp.take_along_axis(x, sort_indices, axis=axis),
                (locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations),
            )

        # random slice at first dimension
        index = jax.random.choice(state.key, locations_on_hypercube.shape[0])
        locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations = jax.tree_util.tree_map(
            lambda x: x[index],
            (locations_on_hypercube, target_concepts_on_hypercube, concept_dist_params_at_locations),
        )

        if jnp.all(jnp.isclose(locations_on_hypercube[:, :, 0], locations_on_hypercube[0, 0, 0])) is False:
            logger.warning("Location slice is not aligned with x_0 axis.")

        # plot
        target_drift = target_concepts_on_hypercube.drift_terms[0]
        drift = concept_dist_params_at_locations.drift_terms_mean[0]

        target_diffusion = target_concepts_on_hypercube.diffusion_terms[0]
        diffusion = concept_dist_params_at_locations.diffusion_terms_mean[0]

        for dim in range(3):
            # get min and max per drift and diffusion, to use same scaling in ground-truth and model
            drifts = jnp.concatenate([target_drift, drift], axis=0)
            drift_min, drift_max = drifts[..., dim].min(), drifts[..., dim].max()
            drift_range = drift_max - drift_min

            diffusions = jnp.concatenate([target_diffusion, diffusion], axis=0)
            diffusion_min, diffusion_max = diffusions[..., dim].min(), diffusions[..., dim].max()
            diffusion_range = diffusion_max - diffusion_min

            axs[0, dim].imshow((target_drift[..., dim] - drift_min) / drift_range, vmin=0, vmax=1)
            axs[1, dim].imshow((drift[..., dim] - drift_min) / drift_range, vmin=0, vmax=1)

            axs[2, dim].imshow((target_diffusion[..., dim] - diffusion_min) / diffusion_range, vmin=0, vmax=1)
            axs[3, dim].imshow((diffusion[..., dim] - diffusion_min) / diffusion_range, vmin=0, vmax=1)

            axs[0, dim].set_title("Dimension " + str(dim))

        axs[0, 0].set_ylabel("Ground-Truth Drift")
        axs[1, 0].set_ylabel("Model Drift")
        axs[2, 0].set_ylabel("Ground-Truth Diffusion")
        axs[3, 0].set_ylabel("Model Diffusion")

        fig.suptitle("Slice at x_0 = " + str(locations_on_hypercube[0, 0, 0]))

        fig_dict = {"3D_vector_field": fig}
        plt.close()

        # plot data paths to check and samples
        if self.plot_observations is True:
            fig_3D = plt.Figure()
            ax_3D = fig_3D.add_axes(111, projection="3d")

            fig_3D.suptitle("Observation Paths")

            obs_config = {"c": "red", "linewidth": self.paths_linewidth}
            samples_config = {"c": "black", "linewidth": self.paths_linewidth}
            init_state_config = {"color": "blue", "s": 3}

            plot_3D_observations_and_samples(
                self,
                state,
                ax_3D,
                obs_times,
                obs_values,
                obs_mask,
                paths_mask,
                locations_on_hypercube,
                fine_delta_x,
                # normalized_init_cond_dist_params,
                init_cond_dist_params,
                obs_config,
                samples_config,
                init_state_config,
            )

            fig_3D.legend(ncol=4)
            fig_dict.update({"3D_Paths": fig_3D})

        return state, fig_dict

