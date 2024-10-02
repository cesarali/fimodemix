import h5py
import numpy as np
import jax.numpy as jnp

def get_np_sde_data(file_name):
    data = {}
    try:
        with h5py.File(file_name, 'r') as f:
            for k in f.keys():
                try:
                    # Check if the data is an array or a scalar
                    if len(f[k].shape) == 0:
                        data[k] = f[k][()]
                    else:
                        # Numpy uses C order, so we need to transpose the arrays
                        data[k] = np.array(f[k][:]).T
                except Exception as e:
                    print(f"Error processing key {k}: {e}")
    except Exception as e:
        print(f"Error opening file {file_name}: {e}")
    return data

# d = get_np_sde_data("data/sde_data_langevin.h5")
        