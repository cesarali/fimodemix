import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from fimodemix.utils.grids import define_mesh_points

@dataclass
class FIMSDEpDatabatch:
    obs_values: torch.Tensor
    obs_times: torch.Tensor

    diffusion_at_hypercube: torch.Tensor
    drift_at_hypercube: torch.Tensor
    hypercube_locations: torch.Tensor

    diffusion_parameters: torch.Tensor
    drift_parameters: torch.Tensor
    process_label:torch.Tensor
    process_dimension:torch.Tensor
    
    #init_condition_distr_parameters: torch.Tensor = None
    #f_strs: torch.Tensor = None
    #g_strs: torch.Tensor = None

# Euler-Maruyama method for integrating SDEs
def euler_maruyama_step(states,dt,drift_function,diffusion_function,drift_params,diffusion_params):
    # Calculate the deterministic part
    drift = drift_function(states, None, drift_params)
    diffusion = diffusion_function(states,None,diffusion_params)
    # Update the state with the deterministic part
    new_states = states + drift * dt
    # Add the diffusion part
    new_states += diffusion * torch.sqrt(torch.tensor(dt)) * torch.randn_like(states)
    return new_states

# Lorenz 63 model parameters
def lorenz63(states,time,params):
    sigma = params[:, 0]
    beta = params[:, 1]
    rho = params[:, 2]
    
    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return torch.stack([dxdt, dydt, dzdt], dim=1)

def constant_diffusion(states,time,params):
    """here we assume independent diffusion, this will be the diagonal of the Diffusion matrix"""
    assert params.shape == states.shape
    return params.to(states.device)

# Drift function for the damped cubic oscillator
def damped_cubic_oscillator_drift(states,time,params):
    damping_coeff = params[:, 0]  # Damping coefficient
    alpha_coeff = params[:, 1]  # Coefficient for x1^3
    beta_coeff = params[:, 2]  # Coefficient for x2^3

    x1 = states[:, 0]
    x2 = states[:, 1]
    
    # Determine the drift terms
    dx1dt = -(damping_coeff * x1**3 - alpha_coeff * x2**3)
    dx2dt = -(beta_coeff * x1**3 + damping_coeff * x2**3)
    
    # Combine drift terms into a vector
    return torch.cat((dx1dt.unsqueeze(1), dx2dt.unsqueeze(1)), dim=1)

# Generate paths from the Lorenz 63 model with diffusion
def generate_lorenz_paths(num_paths, num_steps, dt):
    """creates samples for the parameters then samples the processes"""
    # Define distributions for parameters
    sigma_dist = torch.distributions.uniform.Uniform(10, 20)
    beta_dist = torch.distributions.uniform.Uniform(2.5, 5)
    rho_dist = torch.distributions.uniform.Uniform(28, 40)

    diffusion_params = torch.full((num_paths,3),1.)

    # Sample parameters for all paths
    sigma_samples = sigma_dist.sample((num_paths,))
    beta_samples = beta_dist.sample((num_paths,))
    rho_samples = rho_dist.sample((num_paths,))

    drift_params = torch.stack([sigma_samples, beta_samples, rho_samples], dim=1)

    # Initialize states for all paths
    #states = torch.full((num_paths, 3), 1.0)  # Initial conditions (x, y, z)
    states = torch.nn.functional.sigmoid(torch.normal(0., 1., size=(num_paths, 3)))

    # Store paths
    paths = torch.zeros((num_paths, num_steps + 1, 3))  # +1 for initial state
    paths[:, 0] = states.clone()  # Store initial states

    times = torch.linspace(0.,num_steps*dt,num_steps+1)
    times = times[None,:].repeat(num_paths,1)

    # Simulate the paths
    for step in range(num_steps):
        states = euler_maruyama_step(states,dt, lorenz63,constant_diffusion,drift_params,diffusion_params)  # Diffusion term
        paths[:, step + 1] = states.clone()  # Store new states

    return paths,times,drift_params,diffusion_params

# Generate from damped cubic
def generate_damped_cubic_paths(num_paths, num_steps, dt):
    # Define distributions for parameters
    g1_dist = torch.distributions.uniform.Uniform(0.1, 0.5)
    g2_dist = torch.distributions.uniform.Uniform(0.1, 0.5)
    damping_dist = torch.distributions.uniform.Uniform(0.1, 0.5)
    alpha_dist = torch.distributions.uniform.Uniform(1.0, 3.0)
    beta_dist = torch.distributions.uniform.Uniform(0.1, 1.0)

    # Sample parameters for all paths
    g1_samples = g1_dist.sample((num_paths,))
    g2_samples = g2_dist.sample((num_paths,))
    damping_samples = damping_dist.sample((num_paths,))
    alpha_samples = alpha_dist.sample((num_paths,))
    beta_samples = beta_dist.sample((num_paths,))
    
    drift_params = torch.stack([damping_samples, alpha_samples, beta_samples], dim=1)
    diffusion_params = torch.stack([g1_samples, g2_samples], dim=1)
    
    # Initialize states for all paths
    states = torch.full((num_paths, 2), 1.0)  # Initial conditions (x1, x2)

    # Store paths
    paths = torch.zeros((num_paths, num_steps + 1, 2))  # +1 for initial state
    paths[:, 0] = states.clone()  # Store initial states
    times = torch.linspace(0.,num_steps*dt,num_steps+1)
    times = times[None,:].repeat(num_paths,1)

    # Simulate the paths
    for step in range(num_steps):
        # Diffusion term
        states = euler_maruyama_step(states,
                                     dt,
                                     damped_cubic_oscillator_drift,constant_diffusion,
                                     drift_params,diffusion_params)  
        paths[:, step + 1] = states.clone()  # Store new states
    return paths,times,drift_params,diffusion_params

def define_fim_sde_data(
        obs_values,
        obs_times,
        drift_function,
        diffusion_function,
        drift_parameters,
        diffusion_parameters,
        process_label,
        process_dimension,
        num_hypercube_points
        )->FIMSDEpDatabatch:
    """
    Defines hyper cube and evaluates drift and diffusion there
    """
    num_paths = obs_values.size(0)
    dimensions = obs_values.size(2)

    hypercube_locations = define_mesh_points(num_hypercube_points,dimensions)
    num_hypercube_points = hypercube_locations.size(0)
    hypercube_ = hypercube_locations.repeat((num_paths,1))

    drift_params_ = drift_parameters.repeat_interleave(num_hypercube_points,0)
    diffusion_parameters_ = diffusion_parameters.repeat_interleave(num_hypercube_points,0)

    drift_at_hypercube = drift_function(hypercube_,None,drift_params_)
    drift_at_hypercube = drift_at_hypercube.reshape(num_paths,num_hypercube_points,dimensions)
    
    diffusion_at_hypercube = diffusion_function(hypercube_,None,diffusion_parameters_)
    diffusion_at_hypercube = diffusion_at_hypercube.reshape(num_paths,num_hypercube_points,dimensions)
    hypercube_ = hypercube_.reshape(num_paths,num_hypercube_points,dimensions)
    
    data = FIMSDEpDatabatch(
        hypercube_locations=hypercube_,
        obs_times=obs_times,
        obs_values=obs_values,
        diffusion_at_hypercube=diffusion_at_hypercube,
        drift_at_hypercube=drift_at_hypercube,
        diffusion_parameters=diffusion_parameters,
        drift_parameters=drift_parameters,
        process_label=process_label,
        process_dimension=process_dimension
    )
    return data

def generate_data():
    from pathlib import Path
    from fimodemix import data_path
    data_path = Path(data_path)
    split_sizes={"train":5000, "test":500, "validation":500}
    for key,value in split_sizes.items():
        # Parameters
        dt = 0.01
        num_paths = value
        number_of_steps = 128
        num_hypercube_points = 1024
        obs_values, obs_times, drift_params,diffusion_params = generate_lorenz_paths(num_paths, number_of_steps, dt)
        process_label = torch.full((obs_values.size(0),1),0)
        process_dimension = torch.full((obs_values.size(0),1),3)

        fim_sde_data = define_fim_sde_data(
            obs_values,
            obs_times,
            lorenz63,
            constant_diffusion,
            drift_params,
            diffusion_params,
            process_label,
            process_dimension,
            num_hypercube_points
        ) 
        lorenz_path = data_path / "parameters_sde" / "lorenz_{0}.tr".format(key)
        torch.save(fim_sde_data,lorenz_path)

        obs_values, obs_times, drift_parameters,diffusion_parameters = generate_damped_cubic_paths(num_paths, number_of_steps, dt)
        process_label = torch.full((obs_values.size(0),1),1)
        process_dimension = torch.full((obs_values.size(0),1),2)

        fim_sde_data = define_fim_sde_data(
            obs_values,
            obs_times,
            damped_cubic_oscillator_drift,
            constant_diffusion,
            drift_parameters,
            diffusion_parameters,
            process_label,
            process_dimension,
            num_hypercube_points
            )
        damped_path = data_path / "parameters_sde" / "damped_{0}.tr".format(key)
        torch.save(fim_sde_data,damped_path)

if __name__=="__main__":
    num_paths = 200
    number_of_steps = 128
    num_hypercube_points = 1024
    dt = 0.01

    obs_values, obs_times, drift_params,diffusion_params = generate_lorenz_paths(num_paths, number_of_steps, dt)
    process_label = torch.full((obs_values.size(0),1),1)
    fim_sde_data = define_fim_sde_data(
            obs_values,
            obs_times,
            lorenz63,
            constant_diffusion,
            drift_params,
            diffusion_params,
            process_label,
            num_hypercube_points
        )
    
    obs_values, obs_times, drift_parameters,diffusion_parameters = generate_damped_cubic_paths(num_paths, number_of_steps, dt)
    process_label = torch.full((obs_values.size(0),1),1)

    fim_sde_data = define_fim_sde_data(
        obs_values,
        obs_times,
        damped_cubic_oscillator_drift,
        constant_diffusion,
        drift_parameters,
        diffusion_parameters,
        process_label,
        num_hypercube_points
        )
