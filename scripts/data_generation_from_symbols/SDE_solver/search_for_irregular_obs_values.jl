using Base.Threads
using Logging
using Pkg

try
    using HDF5
catch
    Pkg.add("HDF5")
    using HDF5
end

"""
Some code to detect paths with very large differences between two consecutive steps.
"""



names = [
    # "/cephfs_projects/foundation_models/data/SDE_concepts_new/06-08-24/dim-1/1",
    "/home/cvejoski/Projects/FoundationModels/Wiener-Procs-FM/data/dim-1/1"
    # "/cephfs_projects/foundation_models/data/SDE_concepts_new/06-08-24/dim-1/10"
]

function load_data(names)
    data = Dict()
    for name in names
        path = name * "/obs_values.h5"
        h5open(path, "r") do file
            data[name] = read(file["data"])[:, :, :, :]
        end
    end
    return data
end

function detect_step_diffs(data; threshold=10, max_mean_diff_fact=10)
    """
    Detects if the difference between two consecutive steps is greater than a threshold.
    """
    large_diff_paths = Dict()
    for (name, obs_values) in data
        large_diff_paths[name] = Dict()
        for sample in 1:size(obs_values, 1)
            for path in 1:size(obs_values, 2)
                diffs = abs.(obs_values[sample, path, 2:end] - obs_values[sample, path, 1:end-1])
                max_diff = maximum(diffs)
                mean = sum(diffs) / length(diffs)
                if max_diff > threshold && max_diff / mean > max_mean_diff_fact
                    large_diff_paths[name][(sample, path)] = max_diff
                end
            end
        end
    end
    return large_diff_paths
end

function compute_large_path_diff_percentage(data, large_diff_paths)
    """
    Computes the percentage of paths that have a large difference between two consecutive steps.
    """
    large_diff_percentage = Dict(name => 100 * length(large_diff_paths[name]) / (size(data[name], 1) * size(data[name], 2)) for name in keys(data))
    return large_diff_percentage
end

data = load_data(names)
large_diff_paths = detect_step_diffs(data)
large_diff_percentage = compute_large_path_diff_percentage(data, large_diff_paths)
println(large_diff_percentage)
println(large_diff_paths)

# path = names[1] * "/f_strs.h5"
# println(path)
# h5open(path, "r") do file
#    f_strs = read(file["data"])
#    for f_str in f_strs
#        println(f_str)
#    end
# end
# path = names[1] * "/g_strs.h5"
# h5open(path, "r") do file
#    g_strs = read(file["data"])
#    for g_str in g_strs
#        println(g_str)
#    end
# end