function HP1NBD_DE()
    f(u, p, t) = -7.5912*u+10.8822*u^2+43.4492*u^3
    g(u, p, t) = -0.3828*u+2.2461*u^3+0.0585*exp(3u)
    t_start = 0.0
    # t_end = 1
    # dt = 2e-5
    t_end = 50/48
    dt = 1/48000
    num_paths = 1
    initial_condition = 0.08629090909090908
    return f, g, t_start, t_end, dt, num_paths, initial_condition
end

function facebook()
    f(u, p, t) = 0.0829*u
    g(u, p, t) = 0.4039*u
    t_start = 0.0
    t_end = 10
    dt = 1e-4
    num_paths = 1
    initial_condition = 229.09
    return f, g, t_start, t_end, dt, num_paths, initial_condition
end