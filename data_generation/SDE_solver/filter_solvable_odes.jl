
using Pkg
using Base.Threads
using JSON3

try
    using CSV
catch
    Pkg.add("CSV")
    using CSV
end

try
    using Term.Progress
catch
    Pkg.add("Term")
    using Term.Progress
end

try
    using JSON
catch
    Pkg.add("JSON")
    using JSON
end

try
    using ArgParse
catch
    Pkg.add("ArgParse")
    using ArgParse
end

using ProgressBars: ProgressBar as ProgressBarBase




include("helpers.jl")
include("threadsafecollections.jl")

using .ThreadSafeCollections: ThreadSafeSet, ThreadSafeDict, add!, remove!, length

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--input"
        help = "ODEs to solve"
        arg_type = String
        "--num_paths"
        help = "Number of paths to generate"
        arg_type = Int
        default = 300
        "--mean"
        help = "Mean of the initial conditino distribution"
        arg_type = Float32
        default = 0.0
        "--std"
        help = "Standard deviation of the initial condition distribution"
        arg_type = Float32
        default = 1.0
        "--dim"
        help = "Dimension of the ODE"
        arg_type = Int
        default = 1
        "--output"
        help = "Output folder"
        arg_type = String
    end
    return parse_args(s)
end

function read_file(filename)
    lines = Set{Vector{String}}()
    open(filename, "r") do file
        for line in ProgressBarBase(eachline(file))
            if line != ""
                push!(lines, split(line, ','))
            end
        end
    end
    return collect(lines)
end

function is_ode_solvable(ode_ix, odes, initial_condition)
    tspan = (Float32(1.0), Float32(10.0))
    p = (ode_ix, odes)
    prob = ODEProblem(f!, initial_condition, tspan, p, dt=0.014)
    try
        sol = Base.invokelatest(solve, prob, Euler())
        return check_if_ode_is_valid(sol)
    catch e
        @warn "Error in solving ODE: $e"
        return false
    end
end

function f!(du, u, p, t)
    ix = p[1]
    funct = p[2][ix]
    return funct(du, u, p, t)
end

function solvable_odes(odes, str_odes, dim, num_paths, mean, std, output_path, invalid_paths_threshold=5)
    odes_stats = ThreadSafeDict{String,Tuple{Int,Int}}()

    minibatch_size = 500
    num_odes = Base.length(odes)
    num_batches = ceil(Int, num_odes / minibatch_size)
    pbar = ProgressBar(; expand=true, columns=:detailed, colors="#ffffff",
        columns_kwargs=Dict(
            :ProgressColumn => Dict(:completed_char => '█', :remaining_char => '░'),
        ),
    )
    job = addjob!(pbar; N=num_batches, description="Checking solvability of ODEs")
    start!(pbar)
    render(pbar)
    for batch in 1:num_batches
        solvable_odes = ThreadSafeSet{Vector{String}}()
        start_idx = (batch - 1) * minibatch_size + 1
        end_idx = min(batch * minibatch_size, num_odes)
        # job_mini = addjob!(pbar; N=end_idx - start_idx + 1, description="Minibatch $batch")
        function f(ode_ix)
            valid_paths = 0
            invalid_paths = 0
            for _ in 1:num_paths
                initial_condition = zeros(Float32, dim)
                for i in 1:dim
                    initial_condition[i] = rand(Normal(mean, std))
                end
                is_solvable = is_ode_solvable(ode_ix, odes[start_idx:end_idx], initial_condition)
                if is_solvable
                    valid_paths += 1
                else
                    invalid_paths += 1
                end
                if invalid_paths >= invalid_paths_threshold
                    break
                end
            end

            if valid_paths >= num_paths - invalid_paths_threshold
                add!(solvable_odes, str_odes[(batch-1)*minibatch_size+ode_ix])
            end
            return valid_paths, invalid_paths
        end

        Threads.@threads for i in ProgressBarBase(1:end_idx-start_idx+1)
            # for i in ProgressBarBase(1:end_idx-start_idx+1)
            path_stats = f(i)
            add!(odes_stats, join(str_odes[(batch-1)*minibatch_size+i], "; "), path_stats)
        end

        open(output_path, "a") do file
            # @info "Writing $(length(solvable_odes)) solvable ODEs to file: $output_path"
            for ode in solvable_odes.set
                println(file, join(ode, ","))
            end
        end

        json_file = output_path * ".json"
        open(json_file, "w") do file
            write(file, JSON.json(odes_stats.dict))
        end
        update!(job)
        render(pbar)
    end
    stop!(pbar)
    return solvable_odes, odes_stats
end

function main()
    parsed_args = parse_commandline()
    println("Parsed args: ", parsed_args)
    str_odes = read_file(parsed_args["input"])
    dim = parsed_args["dim"]
    mean = parsed_args["mean"]
    std = parsed_args["std"]
    num_paths = parsed_args["num_paths"]
    output_path = parsed_args["output"]

    begin
        compiled_odes = Threads.@spawn compile_functions(str_odes; is_f=true)
        compiled_odes = fetch(compiled_odes)
    end
    solvable_odes(compiled_odes, str_odes, dim, num_paths, mean, std, output_path)
end

main()