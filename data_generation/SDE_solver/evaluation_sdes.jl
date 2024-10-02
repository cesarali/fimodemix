using DifferentialEquations

struct EvalParameters
    name::String
    f::Function
    g::Function
    t_start::Float64
    t_end::Float64
    dt::Float64
    dt_eval::Float64
    num_paths::Int
    initial_condition::Union{Float64, Vector{Float64}}
    max_diffs::Union{Nothing, Vector{Float64}}
end

function get_max_ode_sol_diff(f, u0, tspan; alg=Tsit5())
    """
    Get the difference between the largest and smallest values of the ODE solution.
    """
    prob = ODEProblem(f, u0, tspan)
    sol = solve(prob, alg)
    return [maximum(sol[i,:]) - minimum(sol[i,:]) for i in 1:length(u0)]
end

function ornstein_uhlenbeck_parameters()
    """
    See https://arxiv.org/pdf/2102.03657 section 4.1
    """
    f(u, p, t) = 0.02*t - 0.1*u
    g(u, p, t) = 0.4
    t_start = 0.0
    t_end = 63.0
    dt = 1
    dt_eval = 1
    num_paths = 8192
    initial_condition = 0.0
    return EvalParameters("ornstein_uhlenbeck", f, g, t_start, t_end, dt, dt_eval, num_paths, initial_condition, nothing)
end

function double_well_parameters()
    f(u, p, t) = 4*(u - u^3)
    g(u, p, t) = sqrt(max(4-1.25*u^2, 0.0))
    t_start = 0.0
    t_end = 10.0
    dt = 0.002
    dt_eval = 0.1
    num_paths = 300
    initial_condition = 0.0
    return EvalParameters("double_well", f, g, t_start, t_end, dt, dt_eval, num_paths, initial_condition, nothing)
end

function damped_linear_oscillator_parameters(max_diff_scale=200)
    t_start = 0.0
    t_end = 20.0
    dt = 0.002
    dt_eval = 0.2
    num_paths = 300
    initial_condition = [2.5, -5.]
    f!(du, u, p, t) = begin
        du[1] = -(0.1*u[1] - 2*u[2])
        du[2] = -(2*u[1] + 0.1*u[2])
    end
    max_diff = get_max_ode_sol_diff(f!, initial_condition, (t_start, t_end))
    g!(du, u, p, t) = begin
        for i = 1:length(du)
            du[i] = max_diff[i] / max_diff_scale
        end
    end
    return EvalParameters("damped_linear_oscillator", f!, g!, t_start, t_end, dt, dt_eval, num_paths, initial_condition, max_diff)
end

function damped_cubic_oscillator_parameters(max_diff_scale=200)
    t_start = 0.0
    t_end = 25.0
    dt = 0.002
    dt_eval = 0.25
    num_paths = 300
    initial_condition = [0., -1.]
    f!(du, u, p, t) = begin
        du[1] = -(0.1*u[1]^3 - 2*u[2]^3)
        du[2] = -(2*u[1]^3 + 0.1*u[2]^3)
    end
    max_diff = get_max_ode_sol_diff(f!, initial_condition, (t_start, t_end))
    g!(du, u, p, t) = begin
        for i = 1:length(du)
            du[i] = max_diff[i] / max_diff_scale
        end
    end
    return EvalParameters("damped_cubic_oscillator", f!, g!, t_start, t_end, dt, dt_eval, num_paths, initial_condition, max_diff)
end

function duffing_oscillator_parameters(max_diff_scale=200)
    t_start = 0.0
    t_end = 20.0
    dt = 0.002
    dt_eval = 0.2
    num_paths = 300
    initial_condition = [3., 2.]
    f!(du, u, p, t) = begin
        du[1] = u[2]
        du[2] = -(u[1]^3 - u[1] + 0.35*u[2])
    end
    max_diff = get_max_ode_sol_diff(f!, initial_condition, (t_start, t_end))
    g!(du, u, p, t) = begin
        for i = 1:length(du)
            du[i] = max_diff[i] / max_diff_scale
        end
    end
    return EvalParameters("duffing_oscillator", f!, g!, t_start, t_end, dt, dt_eval, num_paths, initial_condition, max_diff)
end

function selkov_glycolysis_parameters(max_diff_scale=200)
    t_start = 0.0
    t_end = 30.0
    dt = 0.002
    dt_eval = 0.3
    num_paths = 300
    initial_condition = [0.7, 1.25]
    f!(du, u, p, t) = begin
        du[1] = -(u[1] - 0.08*u[2] - u[1]^2*u[2])
        du[2] = (0.6 - 0.08*u[2] - u[1]^2*u[2])
    end
    max_diff = get_max_ode_sol_diff(f!, initial_condition, (t_start, t_end))
    g!(du, u, p, t) = begin
        for i = 1:length(du)
            du[i] = max_diff[i] / max_diff_scale
        end
    end
    return EvalParameters("selkov_glycolysis", f!, g!, t_start, t_end, dt, dt_eval, num_paths, initial_condition, max_diff)
end

function hopf_bifurcation_parameters(max_diff_scale=200)
    t_start = 0.0
    t_end = 20.0
    dt = 0.002
    dt_eval = 0.2
    num_paths = 300
    initial_condition = [2., 2.]
    f!(du, u, p, t) = begin
        du[1] = 0.5*u[1] + u[2] - u[1]*(u[1]^2 + u[2]^2)
        du[2] = -u[1] + 0.5*u[2] - u[2]*(u[1]^2 + u[2]^2)
    end
    max_diff = get_max_ode_sol_diff(f!, initial_condition, (t_start, t_end))
    g!(du, u, p, t) = begin
        for i = 1:length(du)
            du[i] = max_diff[i] / max_diff_scale
        end
    end
    return EvalParameters("hopf_bifurcation", f!, g!, t_start, t_end, dt, dt_eval, num_paths, initial_condition, max_diff)
end

function lorentz_63_parameters(max_diff_scale=200)
    t_start = 0.0
    t_end = 10.0
    dt = 0.002
    dt_eval = 0.1
    num_paths = 300
    initial_condition = [-8., 7., 27.]
    f!(du, u, p, t) = begin
        du[1] = 10*(u[2] - u[1])
        du[2] = u[1]*(28 - u[3]) - u[2]
        du[3] = u[1]*u[2] - (8/3)*u[3]
    end
    max_diff = get_max_ode_sol_diff(f!, initial_condition, (t_start, t_end))
    g!(du, u, p, t) = begin
        for i = 1:length(du)
            du[i] = max_diff[i] / max_diff_scale
        end
    end
    return EvalParameters("lorentz_63", f!, g!, t_start, t_end, dt, dt_eval, num_paths, initial_condition, max_diff)
end

function langevin_parameters()
    t_start = 0.0
    t_end = 1000.0
    dt = 0.04
    dt_eval = 0.04
    num_paths = 1
    initial_condition = 1.0
    f(u, p, t) = -3*u
    g(u, p, t) = 3
    return EvalParameters("langevin", f, g, t_start, t_end, dt, dt_eval, num_paths, initial_condition, nothing)
end

function double_well2_parameters()
    f(u, p, t) = (u - u^3)
    g(u, p, t) = sqrt(1+u^2)
    t_start = 0.0
    t_end = 1000.0
    dt = 0.04
    dt_eval = 0.04
    num_paths = 1
    initial_condition = 1.0
    return EvalParameters("double_well2", f, g, t_start, t_end, dt, dt_eval, num_paths, initial_condition, nothing)
end

function bisde_2D_sde_ground_truth_parameters()
    t_start = 0.0
    t_end = 2.5
    dt = 0.025
    dt_eval = 0.025
    num_paths = 800
    initial_condition = [1., 1.]
    f!(du, u, p, t) = begin
        du[1] = u[1] - u[2] - u[1]*u[2]^2 - u[1]^3
        du[2] = u[1] + u[2] - u[1]^2*u[2] - u[2]^3
    end
    g!(du, u, p, t) = begin
        du[1] = sqrt(1+u[2]^2)
        du[2] = sqrt(1+u[1]^2)
    end
    return EvalParameters("bisde_2D_sde_ground_truth", f!, g!, t_start, t_end, dt, dt_eval, num_paths, initial_condition, nothing)
end

function bisde_2D_sde_model_prediction_parameters()
    t_start = 0.0
    t_end = 2.5
    dt = 0.025
    dt_eval = 0.025
    num_paths = 800
    initial_condition = [1., 1.]
    f!(du, u, p, t) = begin
        du[1] = 0.9941*u[1] - 0.9738*u[2] - 1.0076*u[1]*u[2]^2 - 1.0294*u[1]^3
        du[2] = 0.9883*u[1] + 0.9452*u[2] - 1.0744*u[1]^2*u[2] - 0.9387*u[2]^3
    end
    g!(du, u, p, t) = begin
        du[1] = sqrt(1.0050+0.9897*u[2]^2)
        du[2] = sqrt(0.9984+1.0306*u[1]^2)
    end
    return EvalParameters("bisde_2D_sde_model_prediction", f!, g!, t_start, t_end, dt, dt_eval, num_paths, initial_condition, nothing)
end

# function ornstein_uhlenbeck_varied_theta_1()
#     theta = 0.01
#     theta_prime = 0.05
#     lambd = 1
#     f(u, p, t) = (theta + (1-lambd)*theta_prime)*u
#     g(u, p, t) = 0.2
#     t_start = 0.0
#     t_end = 63.0
#     dt = 1
#     num_paths = 300
#     initial_condition = 0.0
#     return f, g, t_start, t_end, dt, num_paths, initial_condition, nothing
# end

# function ornstein_uhlenbeck_varied_theta_08()
#     theta = 0.01
#     theta_prime = 0.05
#     lambd = 0.8
#     f(u, p, t) = (theta + (1-lambd)*theta_prime)*u
#     g(u, p, t) = 0.2
#     t_start = 0.0
#     t_end = 63.0
#     dt = 1
#     num_paths = 300
#     initial_condition = 0.0
#     return f, g, t_start, t_end, dt, num_paths, initial_condition, nothing
# end

# function ornstein_uhlenbeck_varied_theta_06()
#     theta = 0.01
#     theta_prime = 0.05
#     lambd = 0.6
#     f(u, p, t) = (theta + (1-lambd)*theta_prime)*u
#     g(u, p, t) = 0.2
#     t_start = 0.0
#     t_end = 63.0
#     dt = 1
#     num_paths = 300
#     initial_condition = 0.0
#     return f, g, t_start, t_end, dt, num_paths, initial_condition, nothing
# end

# function ornstein_uhlenbeck_varied_theta_04()
#     theta = 0.01
#     theta_prime = 0.05
#     lambd = 0.4
#     f(u, p, t) = (theta + (1-lambd)*theta_prime)*u
#     g(u, p, t) = 0.2
#     t_start = 0.0
#     t_end = 63.0
#     dt = 1
#     num_paths = 300
#     initial_condition = 0.0
#     return f, g, t_start, t_end, dt, num_paths, initial_condition, nothing
# end

# function ornstein_uhlenbeck_varied_theta_02()
#     theta = 0.01
#     theta_prime = 0.05
#     lambd = 0.2
#     f(u, p, t) = (theta + (1-lambd)*theta_prime)*u
#     g(u, p, t) = 0.2
#     t_start = 0.0
#     t_end = 63.0
#     dt = 1
#     num_paths = 300
#     initial_condition = 0.0
#     return f, g, t_start, t_end, dt, num_paths, initial_condition, nothing
# end

# function ornstein_uhlenbeck_varied_theta_0()
#     theta = 0.01
#     theta_prime = 0.05
#     lambd = 0
#     f(u, p, t) = (theta + (1-lambd)*theta_prime)*u
#     g(u, p, t) = 0.2
#     t_start = 0.0
#     t_end = 63.0
#     dt = 1
#     num_paths = 300
#     initial_condition = 0.0
#     return f, g, t_start, t_end, dt, num_paths, initial_condition, nothing
# end

# function ornstein_uhlenbeck_varied_sigma_1()
#     sigma = 0.2
#     sigma_prime = 1
#     lambd = 1
#     f(u, p, t) = 0.1*u
#     g(u, p, t) = (sigma + (1-lambd)*sigma_prime)
#     t_start = 0.0
#     t_end = 63.0
#     dt = 1
#     num_paths = 300
#     initial_condition = 0.0
#     return f, g, t_start, t_end, dt, num_paths, initial_condition, nothing
# end

# function ornstein_uhlenbeck_varied_sigma_08()
#     sigma = 0.2
#     sigma_prime = 1
#     lambd = 0.8
#     f(u, p, t) = 0.1*u
#     g(u, p, t) = (sigma + (1-lambd)*sigma_prime)
#     t_start = 0.0
#     t_end = 63.0
#     dt = 1
#     num_paths = 300
#     initial_condition = 0.0
#     return f, g, t_start, t_end, dt, num_paths, initial_condition, nothing
# end

# function ornstein_uhlenbeck_varied_sigma_06()
#     sigma = 0.2
#     sigma_prime = 1
#     lambd = 0.6
#     f(u, p, t) = 0.1*u
#     g(u, p, t) = (sigma + (1-lambd)*sigma_prime)
#     t_start = 0.0
#     t_end = 63.0
#     dt = 1
#     num_paths = 300
#     initial_condition = 0.0
#     return f, g, t_start, t_end, dt, num_paths, initial_condition, nothing
# end

# function ornstein_uhlenbeck_varied_sigma_04()
#     sigma = 0.2
#     sigma_prime = 1
#     lambd = 0.4
#     f(u, p, t) = 0.1*u
#     g(u, p, t) = (sigma + (1-lambd)*sigma_prime)
#     t_start = 0.0
#     t_end = 63.0
#     dt = 1
#     num_paths = 300
#     initial_condition = 0.0
#     return f, g, t_start, t_end, dt, num_paths, initial_condition, nothing
# end

# function ornstein_uhlenbeck_varied_sigma_02()
#     sigma = 0.2
#     sigma_prime = 1
#     lambd = 0.2
#     f(u, p, t) = 0.1*u
#     g(u, p, t) = (sigma + (1-lambd)*sigma_prime)
#     t_start = 0.0
#     t_end = 63.0
#     dt = 1
#     num_paths = 300
#     initial_condition = 0.0
#     return f, g, t_start, t_end, dt, num_paths, initial_condition, nothing
# end

# function ornstein_uhlenbeck_varied_sigma_0()
#     sigma = 0.2
#     sigma_prime = 1
#     lambd = 0
#     f(u, p, t) = 0.1*u
#     g(u, p, t) = (sigma + (1-lambd)*sigma_prime)
#     t_start = 0.0
#     t_end = 63.0
#     dt = 1
#     num_paths = 300
#     initial_condition = 0.0
#     return f, g, t_start, t_end, dt, num_paths, initial_condition, nothing
# end

