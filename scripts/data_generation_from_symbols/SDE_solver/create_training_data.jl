using Base.Threads
using Logging
using Pkg

try
    using DifferentialEquations
catch
    Pkg.add("DifferentialEquations")
    using DifferentialEquations
end

try
    using Distributions
catch
    Pkg.add("Distributions")
    using Distributions
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

using Random

# Set the global seed
Random.seed!(1234)

include("helpers.jl")

const num_trajectories = 300
const max_num_samples = 1000 # How many samples we should process
const use_diffusion = true # If false, we set g to zero no matter what the file says
const use_ode_former_filters = true

function read_jsonl(filename)
    @info "Checking if file exists: $filename"
    if !isfile(filename)
        error("File not found: $filename")
    end
    open(filename) do file
        buffer = String[]
        nlines = 0
        for line in eachline(file)
            push!(buffer, line)
            nlines += 1
            if nlines == max_num_samples
                break
            end
        end
        @info "Finished reading file."
        process_buffer(buffer)
    end
end

function read_buffer(buffer)
    f_strs, g_strs, tspans, dts, num_paths, init_conditions, init_condition_distr_parameters = [], [], [], [], [], [], []
    for line in buffer
        json = JSON3.read(line)
        init_distr_parameter_list = [json["init_condition"][2], json["init_condition"][3]]
        init_condition = sample_initial_distribution(init_distr_parameter_list, num_trajectories)
        init_condition_distr_parameter = json_array_to_float32_array(vcat(init_distr_parameter_list[1], init_distr_parameter_list[2]))
        f_str = json["drift"]
        g_str = json["diffusion"]
        if !use_diffusion
            g_str = ["0" for _ in 1:length(f_str)]
        end
        tspan = (Float32(json["grid"]["lower_bound"]), Float32(json["grid"]["upper_bound"]))
        dt = Float32(tspan[2] - tspan[1]) / json["grid"]["grid_size"]
        push!(f_strs, f_str)
        push!(g_strs, g_str)
        push!(tspans, tspan)
        push!(dts, dt)
        push!(num_paths, num_trajectories)
        push!(init_conditions, init_condition)
        push!(init_condition_distr_parameters, init_condition_distr_parameter)
    end
    # Convert tspans, dts, num_paths, init_conditions, init_condition_distr_parameters to Arrays
    tspans = Array{Tuple{Float32,Float32},1}(tspans)
    dts = Array{Float32,1}(dts)
    num_paths = Array{Int32,1}(num_paths)
    init_condition_distr_parameters = list_of_lists_to_2d_array(init_condition_distr_parameters)
    return f_strs, g_strs, tspans, dts, num_paths, init_conditions, init_condition_distr_parameters
end

function solve_sdes_from_strs(f_strs, g_strs, tspans, dts, num_paths, init_conditions; method=SOSRA())
    drift_functions = compile_functions(f_strs; is_f=true)
    diffusion_functions = compile_functions(g_strs; is_f=false)

    num_samples = length(init_conditions)
    grid_size = length(tspans[1][1]:dts[1]:tspans[1][2])
    num_dims = size(init_conditions[1], 2)
    num_locations = 1

    obs_times = Array{Float32,4}(undef, num_samples, num_paths[1], grid_size, 1)
    obs_values = Array{Float32,4}(undef, num_samples, num_paths[1], grid_size, num_dims)
    locations = Array{Float32,3}(undef, num_samples, num_locations, num_dims)
    concepts_at_locations = Array{Float32,3}(undef, num_samples, num_locations, num_dims)
    invalid_sols = fill(false, num_samples)

    # Solve ODEs to get the diffusion constants
    diffusion_constants_list = []
    @time Threads.@threads for i in 1:num_samples
        p = (i, drift_functions, diffusion_functions)
        diffusion_constants = get_diffusion_scaling_constant(init_conditions[i], tspans[i], p)
        if diffusion_constants === nothing
            invalid_sols[i] = true
            continue
        end
        push!(diffusion_constants_list, diffusion_constants)
    end
    @info "Percentage of invalid ODE solutions in batch: $(length(findall(invalid_sols))/length(init_conditions))"
    exit()
    # Rescale g with the diffusion constants.
    diffusion_functions_resc = compile_rescaled_diffusion_functions(g_strs, diffusion_constants_list, invalid_sols)

    @time Threads.@threads for i in 1:num_samples
        if invalid_sols[i]
            continue
        end
        p = (i, drift_functions, diffusion_functions_resc)
        sols = []
        for j in 1:num_paths[i]
            prob = SDEProblem(f!, g!, init_conditions[i][j, :], tspans[i], p)
            try
                sol = Base.invokelatest(solve, prob, method; dt=dts[i], saveat=dts[i]) # We are not using EnsembleProblem because it seems to create a race condition
                if sol.retcode != :Success
                    break
                end
                push!(sols, sol)
            catch e
                @warn "Error in solving SDE: $e"
                break
            end
        end
        if length(sols) != num_paths[i]
            invalid_sols[i] = true
            continue
        end
        obs_times[i, :, :, :] .= sol_to_grid(sols, num_paths[i])
        obs_values[i, :, :, :] .= sol_to_vals(sols, num_paths[i], grid_size, num_dims)
        #TODO Implememt this
        locations[i, :, :] .= zeros(Float32, num_locations, num_dims)
        concepts_at_locations[i, :, :] .= zeros(Float32, num_locations, num_dims)
    end

    invalid_idx = findall(invalid_sols)

    @info "Percentage of total invalid solutions in batch: $(length(invalid_idx)/length(init_conditions))"

    return obs_times, obs_values, locations, concepts_at_locations, invalid_idx
end

function get_ODE_trajectories(init_conditions, tspan, p; method=Euler())
    num_paths = size(init_conditions, 1)
    sols = []
    for j in 1:num_paths
        prob = ODEProblem(f!, init_conditions[j, :], tspan, p, dt=0.01)
        try
            sol = Base.invokelatest(solve, prob, method)
            if !check_if_ode_is_valid(sol)
                return nothing
            end
            push!(sols, sol)
        catch e
            @warn "Error in solving ODE: $e"
            return nothing
        end
    end
    return sols
end

function get_diffusion_scaling_constant(init_conditions, tspan, p)
    """
    Return the diffusion scaling constants.
    Given a diffusion function g, we define the scaling constants so that g_tilde = (g-a)*b+c.
    If the ODE solution is invalid, we return nothing
    """
    num_paths = size(init_conditions, 1)
    num_dims = size(init_conditions, 2)
    sols = get_ODE_trajectories(init_conditions, tspan, p)
    if sols === nothing
        return nothing
    end
    # sols is now of shape [num_paths][num_dims, grid_points]
    @assert length(sols) == num_paths
    @assert size(sols[1]) == (num_dims, length(sols[1].t))

    SNR_upper = rand(Uniform(50, 100), num_dims)
    SNR_lower = similar(SNR_upper)
    for i in 1:num_dims
        SNR_lower[i] = rand(Uniform(1, SNR_upper[i]))
    end
    max_ode_variation = [maximum(maximum(view(sol, d, :)) - minimum(view(sol, d, :)) for sol in sols) for d in 1:num_dims]
    g_upper = max_ode_variation ./ SNR_lower
    g_lower = max_ode_variation ./ SNR_upper

    diffusion_function = p[3][p[1]]
    # Evaluate the diffusion function on the solutions
    g_vals = []
    for i in 1:num_paths
        grid = sols[i].t
        grid_size = length(grid)
        for j in 1:grid_size
            tmp = zeros(Float32, num_dims)
            Base.invokelatest(diffusion_function, tmp, sols[i][:, j], nothing, grid[j])
            push!(g_vals, tmp)
        end
    end
    g_vals = hcat(g_vals...)
    g_max = [maximum(view(g_vals, d, :)) for d in 1:num_dims]
    g_min = [minimum(view(g_vals, d, :)) for d in 1:num_dims]
    a = g_min
    b = similar(a)
    @. b = (g_upper - g_lower) / (g_max - g_min)
    c = g_lower
    return a, b, c
end

function f!(du, u, p, t)
    idx = p[1]
    functions = p[2]
    return functions[idx](du, u, p, t)
end

function g!(du, u, p, t)
    idx = p[1]
    functions = p[3]
    return functions[idx](du, u, p, t)
end

function check_if_ode_is_valid(sol; max_val_threshold=1e100, oscillation_threshold=1e-3, oscillation_discard_probability=0.9)
    if sol.retcode != :Success
        return false
    end
    if use_ode_former_filters
        if maximum(abs.(sol)) > max_val_threshold
            return false
        end
        last_quarter_idx = Int(floor(3 * size(sol, 2) / 4))
        if any(maximum(sol[:, last_quarter_idx:size(sol, 2)]) - minimum(sol[:, last_quarter_idx:size(sol, 2)]) .< oscillation_threshold)
            # Return false with a probability of 90%
            if rand() < oscillation_discard_probability
                return false
            end
        end
    end
    return true
end


function process_buffer(buffer)
    f_strs, g_strs, tspans, dts, num_paths, init_conditions, init_condition_distr_parameters = read_buffer(buffer)
    obs_times, obs_values, locations, concepts_at_locations, invalid_sols = solve_sdes_from_strs(f_strs, g_strs, tspans, dts, num_paths, init_conditions)

    # Remove rows corresponding to invalid solutions
    valid_idx = setdiff(1:length(init_conditions), invalid_sols)
    obs_times = obs_times[valid_idx, :, :, :]
    obs_values = obs_values[valid_idx, :, :, :]
    locations = locations[valid_idx, :, :]
    concepts_at_locations = concepts_at_locations[valid_idx, :, :]
    init_condition_distr_parameters = init_condition_distr_parameters[valid_idx, :]
    store_training_data(obs_times, obs_values, locations, concepts_at_locations, init_condition_distr_parameters)
end

FILENAME = "/cephfs_projects/foundation_models/data/SDE/state_sde/dimension_3/train.jsonl"
@time read_jsonl(FILENAME)