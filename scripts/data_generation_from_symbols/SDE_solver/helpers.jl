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
    using Plots
catch
    Pkg.add("Plots")
    using Plots
end

try
    using Term.Progress
catch
    Pkg.add("Term")
    using Term.Progress
end

using ProgressBars: ProgressBar as ProgressBarBase

try
    using HDF5
catch
    Pkg.add("HDF5")
    using HDF5
end



macro timeout(seconds, expr, fail)
    quote
        tsk = @task $expr
        schedule(tsk)
        Timer($seconds) do timer
            istaskdone(tsk) || Base.throwto(tsk, InterruptException())
        end
        try
            fetch(tsk)
        catch e
            @info "Caught exception: $e"
            $fail
        end
    end
end

function list_of_lists_to_2d_array(list_of_lists, T)
    res = Array{T,2}(undef, Base.length(list_of_lists), Base.length(list_of_lists[1]))
    for i in 1:Base.length(list_of_lists)
        res[i, :] .= list_of_lists[i]
    end
    return res
end

function sol_to_grid(sol, num_paths, grid_size)
    grid = sol[1].t
    # grid has shape [grid_points]
    grid = repeat(grid, 1, num_paths)
    grid = permutedims(grid, (2, 1))
    grid = reshape(grid, size(grid)..., 1)
    downsample_factor = Int(floor(size(grid, 2) / grid_size))
    # grid now has shape [num_paths, grid_points, 1]
    grid = grid[:, 1:downsample_factor:(end-1), :]
    return grid
end

function sol_to_vals(sol, num_paths, num_grid_points, num_dims)
    t = size(sol[1])[2]
    solutions = Array{Float32,4}(undef, num_paths, t, num_dims, 2)
    solutions .= 0.0

    for i in 1:num_paths
        for j in 1:t
            solutions[i, j, :, 1] .= sol[i][j]
            if j < t
                solutions[i, j, :, 2] .= sol[i][j+1] - sol[i][j]
            end
        end
    end
    # solutions now has shape [num_paths, grid_points, num_dims, 2]
    downsample_factor = Int(floor(size(solutions, 2) / num_grid_points))
    solutions = solutions[:, 1:downsample_factor:end-1, :, :]
    return solutions
end

function store_training_data(output_path, obs_times, obs_values, hypercube_locations, drift_functions_at_hypercube, scaled_diffusion_functions_at_hypercube, init_condition_distr_parameters, f_strs, g_strs, patch_id)
    """
    Store training data.
    """
    if !isdir(output_path)
        mkpath(output_path)
    end
    num_dims = size(obs_values, 4)
    path = output_path * "dim-$(string(num_dims))" * "/$patch_id"
    if !isdir(path)
        mkpath(path)
    end
    h5open(path * "/obs_times.h5", "w") do file
        write(file, "data", obs_times)
    end
    h5open(path * "/obs_values.h5", "w") do file
        write(file, "data", obs_values)
    end
    h5open(path * "/hypercube_locations.h5", "w") do file
        write(file, "data", hypercube_locations)
    end
    h5open(path * "/drift_functions_at_hypercube.h5", "w") do file
        write(file, "data", drift_functions_at_hypercube)
    end
    h5open(path * "/scaled_diffusion_functions_at_hypercube.h5", "w") do file
        write(file, "data", scaled_diffusion_functions_at_hypercube)
    end
    h5open(path * "/init_condition_distr_parameters.h5", "w") do file
        write(file, "data", init_condition_distr_parameters)
    end
    h5open(path * "/f_strs.h5", "w") do file
        write(file, "data", f_strs)
    end
    h5open(path * "/g_strs.h5", "w") do file

        write(file, "data", g_strs)
    end
end


function store_training_data(output_path, obs_times, obs_values, hypercube_locations, drift_functions_at_hypercube, init_condition_distr_parameters, f_strs, patch_id)
    """
    Store training data.
    """
    if !isdir(output_path)
        mkpath(output_path)
    end
    num_dims = size(obs_values, 4)
    path = output_path * "dim-$(string(num_dims))" * "/$patch_id"
    if !isdir(path)
        mkpath(path)
    end
    h5open(path * "/obs_times.h5", "w") do file
        write(file, "data", obs_times)
    end
    h5open(path * "/obs_values.h5", "w") do file
        write(file, "data", obs_values)
    end
    h5open(path * "/hypercube_locations.h5", "w") do file
        write(file, "data", hypercube_locations)
    end
    h5open(path * "/drift_functions_at_hypercube.h5", "w") do file
        write(file, "data", drift_functions_at_hypercube)
    end
    h5open(path * "/init_condition_distr_parameters.h5", "w") do file
        write(file, "data", init_condition_distr_parameters)
    end
    h5open(path * "/f_strs.h5", "w") do file
        write(file, "data", f_strs)
    end
end

function python_syntax_to_julia(function_strs)
    """
    Replace symbolic constants with their numerical values.
    """
    function_strs = replace.(function_strs, "**" => "^")
    function_strs = replace.(function_strs, "E" => string(exp(1)))
    function_strs = replace.(function_strs, "euler_gamma" => string(0.5772157))

    for i in 0:5
        function_strs = replace.(function_strs, "x_$i" => "u[$(i+1)]")
    end

    return function_strs
end

# We need to split the function into f and g and define each function with its idx because otherwise this seem to cause namspace issues
function python_functions_to_julia_f(f_strs, idx)
    """
    Based on a list of strings of functions in python syntax, create a julia function with a unique name.
    """
    f_strs = python_syntax_to_julia(f_strs)

    # Parse each expression
    exprs = Meta.parse.(f_strs)

    # Generate the function name dynamically
    func_name = Symbol("f_eval_$(idx)!")


    # Generate the function dynamically
    f_eval! = eval(quote
        function $(func_name)(du, u, p, t)::Float32
            $(Expr(:block, [:(du[$i] = $(exprs[i])) for i in 1:Base.length(exprs)]...))
        end
        $(func_name)
    end)
    return f_eval!
end

function python_functions_to_julia_g(g_strs, idx)
    """
    Based on a list of strings of functions in python syntax, create a julia function with a unique name.
    """
    g_strs = python_syntax_to_julia(g_strs)

    # Parse each expression
    exprs = Meta.parse.(g_strs)

    # Generate the function name dynamically
    func_name = Symbol("g_eval_$(idx)!")

    # Generate the function dynamically
    g_eval! = eval(quote
        function $(func_name)(du, u, p, t)
            $(Expr(:block, [:(du[$i] = $(exprs[i])) for i in 1:Base.length(exprs)]...))
        end
        $(func_name)
    end)
    return g_eval!
end

function python_functions_to_julia_g_resc(g_strs, idx, diffusion_constants)
    """
    Based on a list of strings of functions in python syntax, create a julia function with a unique name.
    """
    g_strs = python_syntax_to_julia(g_strs)

    # Parse each expression
    exprs = Meta.parse.(g_strs)

    a, b, c = diffusion_constants

    # Generate the function name dynamically
    func_name = Symbol("g_eval_resc_$(idx)!")

    # Generate the function dynamically
    g_eval_resc! = eval(quote
        function $(func_name)(du, u, p, t)
            $(Expr(:block, [:(du[$i] = $(exprs[i])) for i in 1:Base.length(exprs)]...))
            @. du = (du - $a) * $b + $c
        end
        $(func_name)
    end)
    return g_eval_resc!
end

function json_array_to_float32_array(json_array)
    return [Float32(x) for x in json_array]
end

function sample_initial_distribution(init_distribution_parameter_list, num_paths)
    num_dims = Base.length(init_distribution_parameter_list[1])
    res = zeros(Float32, num_paths, num_dims)

    for i in 1:num_dims
        res[:, i] .= rand(Normal(init_distribution_parameter_list[1][i], init_distribution_parameter_list[2][i]), num_paths)
    end
    return res
end

function compile_functions(function_strs; is_f=true)
    functions = Vector{Function}()
    pbar = ProgressBar(; expand=true, columns=:detailed, colors="#ffffff",
        columns_kwargs=Dict(
            :ProgressColumn => Dict(:completed_char => '█', :remaining_char => '░'),
        )
    )
    @info "number of functions: $(Base.length(function_strs))"

    job = addjob!(pbar; N=Base.length(function_strs), description="Compiling $(is_f ? "drift" : "diffusion") functions")
    start!(pbar)
    if is_f
        for i in 1:Base.length(function_strs)
            func = python_functions_to_julia_f(function_strs[i], i)
            push!(functions, func)
            update!(job)
            render(pbar)
        end
    else
        for i in 1:Base.length(function_strs)
            func = python_functions_to_julia_g(function_strs[i], i)
            push!(functions, func)
            update!(job)
            render(pbar)
        end
    end
    stop!(pbar)
    return functions
end

function check_if_ode_is_valid(sol; max_val_threshold=100, oscillation_threshold=1e-3, oscillation_discard_probability=0.9, use_ode_former_filters=true)
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

function compile_rescaled_diffusion_functions(g_strs, diff_constants, invalid_sols, f_strs)
    diffusion_functions_resc = Vector{Function}()
    for idx in ProgressBarBase(1:Base.length(g_strs))
        if invalid_sols[idx]
            # @info "Using dummy function for idx: $idx"
            # Just use a dummy function which actually diverges
            # we only do this to maintain the shape
            diffusion_constants = [0.0, 0.0, 0.0]
        else
            diffusion_constants = diff_constants[f_strs[idx][1]*";"*g_strs[idx][1]]
        end
        func = python_functions_to_julia_g_resc(g_strs[idx], idx, diffusion_constants)
        push!(diffusion_functions_resc, func)
    end
    return diffusion_functions_resc
end

function sample_normal_distribution(mean, std)
    dim = size(mean)[1]
    sample = zeros(Float32, dim)
    for i in 1:dim
        sample[i] = rand(Normal(mean[i], std[i]))
    end
    return sample
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

function timeout(f::Function, timeout::Float64, return_on_timeout=nothing)
    task = @async f()
    sleep(timeout)
    if !istaskdone(task)
        @warn "Timeout of function execution!"
        return return_on_timeout
    end
    return fetch(task)
end