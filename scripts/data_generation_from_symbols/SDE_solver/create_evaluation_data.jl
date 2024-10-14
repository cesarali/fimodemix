using Pkg

try
    using DifferentialEquations
catch
    Pkg.add("DifferentialEquations")
    using DifferentialEquations
end

try
    using Plots
catch
    Pkg.add("Plots")
    using Plots
end

try
    using HDF5
catch
    Pkg.add("HDF5")
    using HDF5
end

include("evaluation_sdes.jl")
include("sdes_from_data_driven_paper.jl")

function get_dataset_parameters(names=["lorentz_63"])
    eval_parameters_list = []
    for name in names
        if name == "double_well"
            eval_parameters = double_well_parameters()
        elseif name == "damped_linear_oscillator"
            eval_parameters = damped_linear_oscillator_parameters()
        elseif name == "damped_cubic_oscillator"
            eval_parameters = damped_cubic_oscillator_parameters()
        elseif name == "duffing_oscillator"
            eval_parameters = duffing_oscillator_parameters()
        elseif name == "selkov_glycolysis"
            eval_parameters = selkov_glycolysis_parameters()
        elseif name == "hopf_bifurcation"
            eval_parameters = hopf_bifurcation_parameters()
        elseif name == "lorentz_63"
            eval_parameters = lorentz_63_parameters()
        elseif name == "ornstein_uhlenbeck"
            eval_parameters = ornstein_uhlenbeck_parameters()
        elseif name == "langevin"
            eval_parameters = langevin_parameters()
        elseif name == "double_well2"
            eval_parameters = double_well2_parameters()
        elseif name == "bisde_2D_sde_ground_truth"
            eval_parameters = bisde_2D_sde_ground_truth_parameters()
        elseif name == "bisde_2D_sde_model_prediction"
            eval_parameters = bisde_2D_sde_model_prediction_parameters()
        elseif name == "ornstein_uhlenbeck_varied_sigma_0"
            eval_parameters = ornstein_uhlenbeck_varied_sigma_0()
        elseif name == "ornstein_uhlenbeck_varied_sigma_02"
            eval_parameters = ornstein_uhlenbeck_varied_sigma_02()
        elseif name == "ornstein_uhlenbeck_varied_sigma_04"
            eval_parameters = ornstein_uhlenbeck_varied_sigma_04()
        elseif name == "ornstein_uhlenbeck_varied_sigma_06"
            eval_parameters = ornstein_uhlenbeck_varied_sigma_06()
        elseif name == "ornstein_uhlenbeck_varied_sigma_08"
            eval_parameters = ornstein_uhlenbeck_varied_sigma_08()
        elseif name == "ornstein_uhlenbeck_varied_sigma_1"
            eval_parameters = ornstein_uhlenbeck_varied_sigma_1()
        elseif name == "ornstein_uhlenbeck_varied_theta_0"
            eval_parameters = ornstein_uhlenbeck_varied_theta_0()
        elseif name == "ornstein_uhlenbeck_varied_theta_02"
            eval_parameters = ornstein_uhlenbeck_varied_theta_02()
        elseif name == "ornstein_uhlenbeck_varied_theta_04"
            eval_parameters = ornstein_uhlenbeck_varied_theta_04()
        elseif name == "ornstein_uhlenbeck_varied_theta_06"
            eval_parameters = ornstein_uhlenbeck_varied_theta_06()
        elseif name == "ornstein_uhlenbeck_varied_theta_08"
            eval_parameters = ornstein_uhlenbeck_varied_theta_08()
        elseif name == "ornstein_uhlenbeck_varied_theta_1"
            eval_parameters = ornstein_uhlenbeck_varied_theta_1()
        elseif name == "HP1NBD_DE"
            eval_parameters = HP1NBD_DE()
        elseif name == "facebook"
            eval_parameters = facebook()
        else
            error("Unknown SDE name")
        end
        push!(eval_parameters_list, eval_parameters)
    end
    return eval_parameters_list
end

function solve_sdes(dataset_parameters, method=EM())
    fine_sols = []
    coarse_sols = []
    fine_grids = []
    coarse_grids = []
    for eval_parameters in dataset_parameters
        prob = SDEProblem(eval_parameters.f, eval_parameters.g, eval_parameters.initial_condition, (eval_parameters.t_start, eval_parameters.t_end))
        ensembleprob = EnsembleProblem(prob)
        sol_fine_grid = Base.invokelatest(solve, ensembleprob, method, EnsembleThreads(); dt=eval_parameters.dt, trajectories=eval_parameters.num_paths)

        # Evaluate sol on dt_eval grid
        tspan_coarse = eval_parameters.t_start:eval_parameters.dt_eval:eval_parameters.t_end
        sol_coarse_grid = []

        for sol in sol_fine_grid
            # Interpolate solution on the coarse grid
            interpolated_sol = [sol(t) for t in tspan_coarse]
            push!(sol_coarse_grid, interpolated_sol)
        end
        push!(fine_sols, sol_fine_grid)
        push!(coarse_sols, sol_coarse_grid)
        push!(fine_grids, sol_fine_grid[1].t)
        push!(coarse_grids, tspan_coarse)
    end
    return fine_sols, coarse_sols, fine_grids, coarse_grids
end

function sol_to_vals(sol, num_paths, num_grid_points, num_dims)
    solutions = Array{Float32, 3}(undef, num_paths, num_grid_points, num_dims)
    for i in 1:num_paths
        for j in 1:num_grid_points
            solutions[i, j, :] .= sol[i][j]
        end
    end
    # solutions now has shape [num_paths, grid_points, num_dims]
    return solutions
end

function sols_to_hypercube_locations_and_evaluations(sols, num_points, evaluate_hypercube_on_regular_grid; hypercube_increase_percentage_for_location_grid=0.1)
    hypercube_locations_list = []
    hypercube_drift_values_list, hypercube_diffusion_values_list = [], []
    for sol in sols
        num_paths = length(sol)
        num_dims = length(sol[1][1])
        grid = sol[1].t
        num_grid_points = length(grid)
        obs_values = sol_to_vals(sol, num_paths, num_grid_points, num_dims)
        drift = sol[1].prob.f
        diffusion = sol[1].prob.g
        hypercube_locations = sample_points_in_hypercube(obs_values, num_points, evaluate_hypercube_on_regular_grid; hypercube_increase_percentage_for_location_grid=hypercube_increase_percentage_for_location_grid)
        hypercube_drift_values = evaluate_on_hypercube([drift], hypercube_locations)
        hypercube_diffusion_values = evaluate_on_hypercube([diffusion], hypercube_locations)
        push!(hypercube_locations_list, hypercube_locations)
        push!(hypercube_drift_values_list, hypercube_drift_values)
        push!(hypercube_diffusion_values_list, hypercube_diffusion_values)
    end
    return hypercube_locations_list, hypercube_drift_values_list, hypercube_diffusion_values_list
end

function get_hypercube_grid(lower_bound, upper_bound, num_points_per_dim, evaluate_hypercube_on_regular_grid)
    if lower_bound >= upper_bound
        return zeros(num_points_per_dim)
    end
    if evaluate_hypercube_on_regular_grid
        return range(lower_bound, stop=upper_bound, length=num_points_per_dim)
    end
    return rand(Uniform(lower_bound, upper_bounds), num_points_per_dim)
end

function sample_points_in_hypercube(obs_values, num_points, evaluate_hypercube_on_regular_grid; hypercube_increase_percentage_for_location_grid=0.1)
    """
    Sample random points within a hypercube defined by lower_bounds and upper_bounds.
    """
    obs_values = reshape(obs_values, 1, size(obs_values)...)
    min_per_dimension = minimum(obs_values, dims=(2, 3))[:, 1, 1, :]
    max_per_dimension = maximum(obs_values, dims=(2, 3))[:, 1, 1, :]
    range_per_dimension = max_per_dimension .- min_per_dimension
    lower_bounds = min_per_dimension - (hypercube_increase_percentage_for_location_grid / 2) * range_per_dimension  # [D]
    upper_bounds = max_per_dimension + (hypercube_increase_percentage_for_location_grid / 2) * range_per_dimension  # [D]

    num_samples = size(lower_bounds, 1)
    num_dims = size(lower_bounds, 2)

    num_points_per_dim = Int(round(num_points^(1/num_dims)))
    num_points = num_points_per_dim^num_dims

    res = zeros(Float32, (num_samples, num_points, num_dims))
    for sample in 1:num_samples
        grids = [get_hypercube_grid(lower_bounds[sample, i], upper_bounds[sample, i], num_points_per_dim, evaluate_hypercube_on_regular_grid) for i in 1:num_dims]
        locations =  collect(Iterators.product(grids...))
        res[sample, :, :] = transpose(hcat([collect(loc) for loc in locations]...))
    end

    return res
end

function evaluate_on_hypercube(function_array, hypercube_locations)
    """
    Evaluate the function on the hypercube locations.
    """
    num_samples = 1
    num_dims = size(hypercube_locations, 3)
    res = similar(hypercube_locations)
    for sample in 1:num_samples
        func = function_array[sample]
        for i in 1:size(hypercube_locations, 2)
            if num_dims == 1
                res[sample, i, 1] = Base.invokelatest(func, hypercube_locations[sample, i, 1], nothing, 0.)
            else
                Base.invokelatest(func, view(res, sample, i, :), view(hypercube_locations, sample, i, :), nothing, 0.)
            end
        end
    end

    return res[1, :, :]
end

function add_additive_gaussian_noise(obs_values, max_diffs; noise_percentage=0.1)
    """
    Add Gaussian noise to the observations.
    """
    noise = randn(size(obs_values)) .* noise_percentage .* max_diffs
    obs_values += noise
    return obs_values
end

function get_grids(grid, num_paths)
    grid_len = length(grid)
    # grid has shape [grid_points]
    grid = repeat(grid, 1, num_paths)
    grid = permutedims(grid, (2, 1))
    grid = reshape(grid, num_paths, grid_len, 1)
    # grid now has shape [num_paths, grid_points, 1]
    return grid
end

function store_sols(dataset_parameters, fine_sols, coarse_sols, fine_grids, coarse_grids, hypercube_locations_list, hypercube_drift_values_list, hypercube_diffusion_values_list)
    # Create data directory if it doesn't exist
    if !isdir("wienerfm/data_generation/SDE_solver/data")
        mkdir("wienerfm/data_generation/SDE_solver/data")
    end
    for (i, eval_parameters) in enumerate(dataset_parameters)
        fine_sol, coarse_sol = fine_sols[i], coarse_sols[i]
        fine_grid = get_grids(fine_grids[i], eval_parameters.num_paths)
        coarse_grid = get_grids(coarse_grids[i], eval_parameters.num_paths)
        num_dims = length(eval_parameters.initial_condition)
        fine_values = sol_to_vals(fine_sol, eval_parameters.num_paths, length(fine_sol[1]), num_dims)
        coarse_values = sol_to_vals(coarse_sol, eval_parameters.num_paths, length(coarse_sol[1]), num_dims)
        initial_condition = eval_parameters.initial_condition
        hypercube_locations = hypercube_locations_list[i]
        hypercube_drift_values = hypercube_drift_values_list[i]
        hypercube_diffusion_values = hypercube_diffusion_values_list[i]
        h5open("wienerfm/data_generation/SDE_solver/data/$(eval_parameters.name).h5", "w") do file
            write(file, "fine_values", fine_values)
            write(file, "coarse_values", coarse_values)
            write(file, "fine_grid", fine_grid)
            write(file, "coarse_grid", coarse_grid)
            write(file, "initial_condition", initial_condition)
            write(file, "hypercube_locations", hypercube_locations)
            write(file, "hypercube_drift_values", hypercube_drift_values)
            write(file, "hypercube_diffusion_values", hypercube_diffusion_values)
        end
        if eval_parameters.max_diffs !== nothing
            h5open("wienerfm/data_generation/SDE_solver/data/$(eval_parameters.name)_gaussian_noise.h5", "w") do file
                # Reshape max_diffs to (1, 1, 2) to match the third dimension of obs_values
                reshaped_max_diffs = reshape(eval_parameters.max_diffs, 1, 1, num_dims)
                gaussian_noise_obs_values = add_additive_gaussian_noise(coarse_values, reshaped_max_diffs)
                write(file, "coarse_values", gaussian_noise_obs_values)
                write(file, "coarse_grid", coarse_grid)
                write(file, "initial_condition", initial_condition)
                write(file, "hypercube_locations", hypercube_locations)
                write(file, "hypercube_drift_values", hypercube_drift_values)
                write(file, "hypercube_diffusion_values", hypercube_diffusion_values)
            end
        end
    end
end


names = ["ornstein_uhlenbeck", "double_well", "damped_linear_oscillator", "damped_cubic_oscillator", "duffing_oscillator", "selkov_glycolysis", "hopf_bifurcation", "lorentz_63", "langevin", "double_well2", "bisde_2D_sde_ground_truth", "bisde_2D_sde_model_prediction"]
# names = ["ornstein_uhlenbeck_varied_sigma_0", "ornstein_uhlenbeck_varied_sigma_02", "ornstein_uhlenbeck_varied_sigma_04", "ornstein_uhlenbeck_varied_sigma_06", "ornstein_uhlenbeck_varied_sigma_08", "ornstein_uhlenbeck_varied_sigma_1", "ornstein_uhlenbeck_varied_theta_0", "ornstein_uhlenbeck_varied_theta_02", "ornstein_uhlenbeck_varied_theta_04", "ornstein_uhlenbeck_varied_theta_06", "ornstein_uhlenbeck_varied_theta_08", "ornstein_uhlenbeck_varied_theta_1"]
# names = ["facebook"]
dataset_parameters = get_dataset_parameters(names)
fine_sols, coarse_sols, fine_grids, coarse_grids = solve_sdes(dataset_parameters)

num_hypercube_locations_per_dim = 1000
hypercube_locations_list, hypercube_drift_values_list, hypercube_diffusion_values_list = sols_to_hypercube_locations_and_evaluations(fine_sols, num_hypercube_locations_per_dim, true)

store_sols(dataset_parameters, fine_sols, coarse_sols, fine_grids, coarse_grids, hypercube_locations_list, hypercube_drift_values_list, hypercube_diffusion_values_list)

# # Sols now has shape
# # [num_sdes, num_dims, grid_points, num_paths]
# sde_idx = 1
# plot(sols[sde_idx][1,:,1], lw=2, alpha=0.5, legend=false)
# for path in 2:3
#     plot!(sols[sde_idx][1,:,path], lw=2, alpha=0.5)
# end
# plot!(xlabel="Time", ylabel="State", title="SDE Paths")

# #Save plot as png
# savefig("sde_paths.png")

