import matplotlib.pyplot as plt
import numpy as np
import torch

def select_process(drift_in_grid,diffusion_in_grid,databatch,process_index = 0):
    """select 2d detach and select from index"""
    estimated_drift = drift_in_grid
    estimated_diffusion = diffusion_in_grid

    where_2d = torch.where(databatch.process_dimension == 2)[0]
    if len(where_2d) > 0:
        xy_hypercube = databatch.hypercube_locations[where_2d]
        real_drift = databatch.drift_at_hypercube[where_2d]
        real_diffusion = databatch.diffusion_at_hypercube[where_2d]
        estimated_drift = estimated_drift[where_2d]
        estimated_diffusion = estimated_diffusion[where_2d]

        # Move tensors to CPU and detach them to avoid gradient tracking
        xy_hypercube = xy_hypercube.cpu().detach()
        real_drift = real_drift.cpu().detach()
        real_diffusion = real_diffusion.cpu().detach()
        estimated_drift = estimated_drift.cpu().detach()
        estimated_diffusion = estimated_diffusion.cpu().detach()
        
        return (xy_hypercube[process_index],
                real_drift[process_index],
                real_diffusion[process_index],
                estimated_drift[process_index],
                estimated_diffusion[process_index])
    else:
        return None
            
def plot_drift_diffussion(xy_hypercube, real_drift, real_diffusion, estimated_drift, estimated_diffusion, show=False):
    """Plots estimated drift and diffusion along real parts and returns the figure."""
    # Extract grid points (x, y)
    x, y = xy_hypercube[:, 0], xy_hypercube[:, 1]

    # Real vector fields
    u_real_drift, v_real_drift = real_drift[:, 0], real_drift[:, 1]
    u_real_diffusion, v_real_diffusion = real_diffusion[:, 0], real_diffusion[:, 1]

    # Estimated vector fields
    u_estimated_drift, v_estimated_drift = estimated_drift[:, 0], estimated_drift[:, 1]
    u_estimated_diffusion, v_estimated_diffusion = estimated_diffusion[:, 0], estimated_diffusion[:, 1]

    # Create a figure and 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Plot real drift
    axs[0, 0].quiver(x, y, u_real_drift, v_real_drift)
    axs[0, 0].set_title('Real Drift')

    # Plot real diffusion
    axs[0, 1].quiver(x, y, u_real_diffusion, v_real_diffusion)
    axs[0, 1].set_title('Real Diffusion')

    # Plot estimated drift
    axs[1, 0].quiver(x, y, u_estimated_drift, v_estimated_drift)
    axs[1, 0].set_title('Estimated Drift')

    # Plot estimated diffusion
    axs[1, 1].quiver(x, y, u_estimated_diffusion, v_estimated_diffusion)
    axs[1, 1].set_title('Estimated Diffusion')

    # Adjust layout
    plt.tight_layout()

    if show:
        plt.show()

    return fig  # Return the figure to log
