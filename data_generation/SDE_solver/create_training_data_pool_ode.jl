using Base.Threads
using Logging
using Pkg

try
    using ArgParse
catch
    Pkg.add("ArgParse")
    using ArgParse
end

try
    using JSON3
catch
    Pkg.add("JSON3")
    using JSON3
end

try
    using HDF5
catch
    Pkg.add("HDF5")
    using HDF5
end

try
    using GZip
catch
    Pkg.add("GZip")
    using GZip
end

try
    using ProgressBars: ProgressBar as ProgressBarBase
catch
    Pkg.add("ProgressBars")
    using ProgressBars: ProgressBar as ProgressBarBase
end

using Random

include("threadsafecollections.jl")
using .ThreadSafeCollections: ThreadSafeList, ThreadSafeDict, ThreadSafeSet, add!, get, remove!, length

# Set the global seed
Random.seed!(1234)
include("helpers.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--input"
        help = "ODEs to solve"
        arg_type = String
        default = "C:/Users/cesar/Desktop/Projects/FoundationModels/fimodemix/data/state_sde_full/expressions_3d/dimension_1/state_sde.jsonl.gz"
        "--num_paths"
        help = "Number of paths to generate"
        arg_type = Int
        default = 2
        "--output"
        help = "Output folder"
        arg_type = String
        default = "C:/Users/cesar/Desktop/Projects/FoundationModels/fimodemix/data/state_sde_full/expressions_3d/dimension_1/"
        "--skip"
        help = "Number of examples to skip"
        arg_type = Int
        default = 0
        "--num_samples"
        help = "Number of samples to process"
        arg_type = Int
        default = 1000
        "--use_diffusion"
        help = "Whether to use the diffusion term"
        arg_type = Bool
        default = true
        "--use_ode_former_filters"
        help = "Whether to use the ODE former filters"
        arg_type = Bool
        default = true
        "--hypercube_increase_percentage"
        help = "Percentage to increase the hypercube for location grid"
        arg_type = Float32
        default = 0.1
        "--num_hypercube_locations_per_dim"
        help = "Number of hypercube locations per dimension"
        arg_type = Int
        default = 1000
        "--evaluate_hypercube_on_regular_grid"
        help = "Whether to evaluate the hypercube on a regular grid"
        arg_type = Bool
        default = false
        "--invalid_paths_threshold"
        help = "Max number of invalid paths to skip"
        arg_type = Int
        default = 5
    end
    return parse_args(s)
end

function get_hypercube_grid(lower_bound, upper_bound, num_points_per_dim, evaluate_hypercube_on_regular_grid)
    if lower_bound >= upper_bound
        return zeros(num_points_per_dim)
    end
    if evaluate_hypercube_on_regular_grid
        return range(lower_bound, stop=upper_bound, length=num_points_per_dim)
    end
    return sort(rand(Uniform(lower_bound, upper_bound), num_points_per_dim))
end

function sample_points_in_hypercube(lower_bounds, upper_bounds, num_points, evaluate_hypercube_on_regular_grid)
    """
    Sample random points within a hypercube defined by lower_bounds and upper_bounds.
    """
    num_samples = size(lower_bounds, 1)
    num_dims = size(lower_bounds, 2)
    num_points_per_dim = Int(round(num_points^(1 / num_dims)))
    num_points = num_points_per_dim^num_dims
    res = zeros(Float32, (num_samples, num_points, num_dims))
    @info "Sampling points in hypercube"

    Threads.@threads for sample in ProgressBarBase(1:num_samples)
        grids = [get_hypercube_grid(lower_bounds[sample, i], upper_bounds[sample, i], num_points_per_dim, evaluate_hypercube_on_regular_grid) for i in 1:num_dims]
        locations = collect(Iterators.product(grids...))
        res[sample, :, :] = transpose(hcat([collect(loc) for loc in locations]...))
    end

    return res
end

function evaluate_on_hypercube(function_array, hypercube_locations)
    """
    Evaluate the function on the hypercube locations.
    """
    num_samples = size(hypercube_locations, 1)
    res = similar(hypercube_locations)
    Threads.@threads for sample in ProgressBarBase(1:num_samples)
        try
            func = function_array[sample]
            for i in 1:size(hypercube_locations, 2)
                Base.invokelatest(func, view(res, sample, i, :), view(hypercube_locations, sample, i, :), nothing, nothing)
            end
        catch e
            fill!(view(res, sample, :, :), NaN)
            break
        end
    end

    return res
end

function get_ODE_trajectories(num_paths, init_conditions_params, tspan, p; method=Euler(), dts=nothing, invalid_paths_threshold=5)
    sols = []
    total_paths = 0
    invalid_paths = 0
    while total_paths < num_paths
        initial_condition = sample_normal_distribution(init_conditions_params[1], init_conditions_params[2])
        prob = ODEProblem(f!, initial_condition, tspan, p)
        
        try
            # If dts is provided, use it for both dt and saveat, otherwise use default dt
            if dts != nothing
                sol = Base.invokelatest(solve, prob, method; dt=dts, saveat=dts)
            else
                sol = Base.invokelatest(solve, prob, method; dt=0.01)
            end

            if !check_if_ode_is_valid(sol)
                invalid_paths += 1
                if invalid_paths >= invalid_paths_threshold
                    @warn "Skipping ODE solution. Number of valid paths: $total_paths, number of invalid paths: $invalid_paths"
                    return nothing
                end
                continue
            end
            
            push!(sols, sol)
            total_paths += 1
        catch e
            invalid_paths += 1
            if invalid_paths >= invalid_paths_threshold
                @warn "Skipping ODE solution. Number of valid paths: $total_paths, number of invalid paths: $invalid_paths"
                return nothing
            end
            continue
        end
    end
    return sols
end

function load_data(filename, skip_examples, max_num_samples)
    @info "Checking if file exists: $filename"
    if !isfile(filename)
        error("File not found: $filename")
    end
    pbar = ProgressBar(; expand=true, columns=:detailed, colors="#ffffff",
        columns_kwargs=Dict(
            :ProgressColumn => Dict(:completed_char => '█', :remaining_char => '░'),
        ),
    )
    job = addjob!(pbar; N=max_num_samples, description="Reading File")
    start!(pbar)
    render(pbar)
    GZip.open(filename) do file
        buffer = String[]
        nlines = 0
        skiplines = 0
        for line in eachline(file)
            if skiplines < skip_examples
                skiplines += 1
                continue
            end
            push!(buffer, line)
            nlines += 1
            update!(job)
            render(pbar)
            if nlines == max_num_samples
                break
            end
        end
        return buffer
    end
end

function prepare_data(data, use_diffusion=true, num_trajectories=300)
    # f_strs = ThreadSafeList{JSON3.Array}()
    # tspans = ThreadSafeList{Tuple{Float32,Float32}}()
    # dts = ThreadSafeList{Float32}()
    # num_paths = ThreadSafeList{Int32}()
    # init_condition_distr_parameters = ThreadSafeList{Tuple{Vector{Float32},Vector{Float32}}}()
    f_strs, tspans, dts, dts_eval, num_paths, init_condition_distr_parameters = [], [], [], [], [], []

    @info "Preparing data for solving the SDEs"
    # Threads.@threads for line in ProgressBarBase(data)
    for line in ProgressBarBase(data)
        expression_sample = JSON3.read(line)
        init_distr_parameter_list = expression_sample["init_condition"]

        f_str = expression_sample["drift"]

        tspan = (Float32(expression_sample["grid"]["lower_bound"]), Float32(expression_sample["grid"]["upper_bound"]))
        dt = Float32(tspan[2] - tspan[1]) / expression_sample["grid"]["grid_size"]
        dt_eval = Float32(tspan[2] - tspan[1]) / expression_sample["grid"]["grid_size_eval"]
        # init_dist = json_array_to_float32_array(vcat(init_distr_parameter_list[1], init_distr_parameter_list[2]))
        push!(f_strs, f_str)
        push!(tspans, tspan)
        push!(dts, dt)
        push!(dts_eval, dt_eval)
        push!(num_paths, num_trajectories)
        push!(init_condition_distr_parameters, init_distr_parameter_list)
    end

    # Convert tspans, dts, num_paths, init_conditions, init_condition_distr_parameters to Arrays
    tspans = Array{Tuple{Float32,Float32},1}(tspans)
    dts = Array{Float32,1}(dts)
    num_paths = Array{Int32,1}(num_paths)

    # init_condition_distr_parameters = list_of_lists_to_2d_array(init_condition_distr_parameters)
    return f_strs, tspans, dts, dts_eval, num_paths, init_condition_distr_parameters
end

function compile_functs(f_strs)
    begin
        drift_functions = Threads.@spawn compile_functions(f_strs; is_f=true)
        drift_functions = fetch(drift_functions)
    end
    return drift_functions
end

function solve_odes(num_samples, drift_functions, tspans, dts, dts_eval, num_paths, init_conditions_params; method=EM(), invalid_paths_threshold=5)
    invalid_sols = fill(false, num_samples)

    grid_size = Int((tspans[1][2]-tspans[1][1]) / dts_eval[1])
    finegrid_size = Int((tspans[1][2]-tspans[1][1]) / dts[1])
    num_dims = size(init_conditions_params[1][1])[1]
    obs_times = Array{Float32,4}(undef, num_samples, num_paths[1], grid_size, 1)
    obs_values = Array{Float32,5}(undef, num_samples, num_paths[1], grid_size, num_dims, 2)
    minibatch_size = min(500, num_samples)
    num_batches = ceil(Int, num_samples / minibatch_size)
    for batch in ProgressBarBase(1:num_batches)
        start_idx = (batch - 1) * minibatch_size + 1
        end_idx = min(batch * minibatch_size, num_samples)
        #Threads.@threads 
        for i in ProgressBarBase(1:end_idx-start_idx+1)
            idx = (batch - 1) * minibatch_size + i
            p = (i, drift_functions[start_idx:end_idx])
            #sols = get_SDE_trajectories(num_paths[idx], init_conditions_params[idx], tspans[idx], dts[idx], dts_eval[idx], p, method=method, invalid_paths_threshold=invalid_paths_threshold)
            sols = get_ODE_trajectories(num_paths[idx], init_conditions_params[idx], tspans[idx], p, method=Euler(),dts=dts[idx], invalid_paths_threshold=invalid_paths_threshold)
            if sols === nothing || Base.length(sols) != num_paths[idx] || sols === "timeout"
                invalid_sols[idx] = true
                @info "Inavlid SDE solution $idx"
                continue
            end
            obs_times[idx, :, :, :] .= sol_to_grid(sols, num_paths[idx], grid_size)
            obs_values[idx, :, :, :, :] .= sol_to_vals(sols, num_paths[idx], grid_size, num_dims)
        end
    end
    return obs_times, obs_values, invalid_sols
end

function solve_odes_from_strs(f_strs, tspans, dts, dts_eval, num_paths, num_samples, init_conditions_params; method=EM(), invalid_paths_threshold=5, hypercube_increase_percentage_for_location_grid=0.1, num_hypercube_locations_per_dim=10, evaluate_hypercube_on_regular_grid=false)
    drift_functions = compile_functs(f_strs)

    GC.gc()
    @info "Solving SDEs"
    obs_times, obs_values, invalid_sols = solve_odes(num_samples, drift_functions, tspans, dts, dts_eval, num_paths, init_conditions_params, method=method, invalid_paths_threshold=invalid_paths_threshold)
    GC.gc()

    invalid_idx = findall(invalid_sols)
    valid_idx = setdiff(1:Base.length(init_conditions_params), invalid_idx)
    num_valid = Base.length(valid_idx)
    num_dims = size(init_conditions_params[1][1])[1]
    # Remove invalid solutions
    obs_times = obs_times[valid_idx, :, :, :]
    obs_values = obs_values[valid_idx, :, :, :, :]
    init_conditions_params = init_conditions_params[valid_idx, :]
    drift_functions = drift_functions[valid_idx]
    f_strs = f_strs[valid_idx]

    @info "Percentage of total invalid SDEs in batch: $(1 - num_valid/num_samples)"

    # Evaluate the functions in a hypercube that is slightly larger than the sampled paths
    min_per_dimension = minimum(obs_values, dims=(2, 3))[:, 1, 1, :, 1]
    max_per_dimension = maximum(obs_values, dims=(2, 3))[:, 1, 1, :, 1]
    range_per_dimension = max_per_dimension .- min_per_dimension
    hypercube_min_per_dimension = min_per_dimension - (hypercube_increase_percentage_for_location_grid / 2) * range_per_dimension  # [D]
    hypercube_max_per_dimension = max_per_dimension + (hypercube_increase_percentage_for_location_grid / 2) * range_per_dimension  # [D]
    @assert size(hypercube_min_per_dimension) == (num_valid, num_dims)
    @assert size(hypercube_max_per_dimension) == (num_valid, num_dims)
    hypercube_locations = sample_points_in_hypercube(hypercube_min_per_dimension, hypercube_max_per_dimension, num_hypercube_locations_per_dim, evaluate_hypercube_on_regular_grid)

    @info "Evaluating drift functions on hypercube locations"
    drift_functions_at_hypercube = evaluate_on_hypercube(drift_functions, hypercube_locations)

    # Remove evaluations that contain infs and nans
    non_nan_inf_idx = []
    for i in 1:num_valid
        if !any(isinf.(drift_functions_at_hypercube[i, :, :])) && !any(isnan.(drift_functions_at_hypercube[i, :, :]))
            push!(non_nan_inf_idx, i)
        end
    end

    obs_times = obs_times[non_nan_inf_idx, :, :, :]
    obs_values = obs_values[non_nan_inf_idx, :, :, :, :]
    hypercube_locations = hypercube_locations[non_nan_inf_idx, :, :]
    drift_functions_at_hypercube = drift_functions_at_hypercube[non_nan_inf_idx, :, :]
    init_conditions_params = init_conditions_params[non_nan_inf_idx, :]
    f_strs_out = f_strs[non_nan_inf_idx]
    GC.gc()

    @info "Percentage of total invalid evaluations in batch: $(1 - Base.length(non_nan_inf_idx)/num_samples)"
    @info "Generated $(Base.length(non_nan_inf_idx)) valid samples out of $(num_samples) samples."

    return obs_times, obs_values, hypercube_locations, drift_functions_at_hypercube, init_conditions_params, f_strs_out
end

function main()
    args = parse_commandline()
    filename = args["input"]
    output_path = args["output"]
    num_trajectories = args["num_paths"]
    skip_examples = args["skip"]
    max_num_samples = args["num_samples"]
    use_diffusion = args["use_diffusion"]
    use_ode_former_filters = args["use_ode_former_filters"]
    evaluate_hypercube_on_regular_grid = args["evaluate_hypercube_on_regular_grid"]
    invalid_paths_threshold = args["invalid_paths_threshold"]
    hypercube_increase_percentage_for_location_grid = args["hypercube_increase_percentage"]
    num_hypercube_locations_per_dim = args["num_hypercube_locations_per_dim"]

    data = load_data(filename, skip_examples, max_num_samples)

    f_strs, tspans, dts, dts_eval, num_paths, init_condition_distr_parameters = prepare_data(data, use_diffusion, num_trajectories)
    minibatch_size = min(5_000, max_num_samples)
    start_batch = ceil(Int, skip_examples / minibatch_size) + 1
    num_batches = ceil(Int, max_num_samples / minibatch_size)
    p_bar = ProgressBar(; expand=true, columns=:detailed, colors="#ffffff",
        columns_kwargs=Dict(
            :ProgressColumn => Dict(:completed_char => '█', :remaining_char => '░'),
        ),
    )
    job = addjob!(p_bar; N=num_batches, description="Solving SDEs")
    start!(p_bar)
    render(p_bar)
    for i in start_batch:(num_batches+start_batch-1)
        start_idx = (i - 1) * minibatch_size + 1
        end_idx = min(i * minibatch_size, max_num_samples)
        num_samples = end_idx - start_idx + 1
        f_strs_batch = f_strs[start_idx:end_idx]
        tspans_batch = tspans[start_idx:end_idx]
        dts_batch = dts[start_idx:end_idx]
        dts_eval_batch = dts_eval[start_idx:end_idx]
        num_paths_batch = num_paths[start_idx:end_idx]

        init_condition_distr_parameters_batch = init_condition_distr_parameters[start_idx:end_idx]
        obs_times, obs_values, hypercube_locations, drift_functions_at_hypercube, init_condition_distr_parameters_out, f_strs_out = solve_odes_from_strs(f_strs_batch, tspans_batch, dts_batch, dts_eval_batch, num_paths_batch, num_samples, init_condition_distr_parameters_batch, invalid_paths_threshold=invalid_paths_threshold, hypercube_increase_percentage_for_location_grid=hypercube_increase_percentage_for_location_grid, num_hypercube_locations_per_dim=num_hypercube_locations_per_dim, evaluate_hypercube_on_regular_grid=evaluate_hypercube_on_regular_grid)

        init_condition_distr_parameters_out = [json_array_to_float32_array(vcat(x[1], x[2])) for x in init_condition_distr_parameters_out]
        init_condition_distr_parameters_out = list_of_lists_to_2d_array(init_condition_distr_parameters_out, Float32)
        f_strs_out = list_of_lists_to_2d_array(f_strs_out, String)
        store_training_data(output_path, obs_times, obs_values, hypercube_locations, drift_functions_at_hypercube, init_condition_distr_parameters_out, f_strs_out, i)                            
        update!(job)
    end
    stop!(p_bar)
end

main()