import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from fimodemix.utils.grids import define_mesh_points

from fimodemix.data.datasets import (
    FIMCompartementsDatabatch,
    FIMCompartementsDatabatchTuple
)

import abc
import numpy as np
from scipy.integrate import odeint
from collections import namedtuple
from scipy.integrate import odeint
from dataclasses import dataclass, field

# Define the namedtuple
FIMCompartementsDatabatchTuple = namedtuple(
    'FIMCompartementsDatabatchTuple',
    [
        'obs_values', 'obs_times',
        'dosing_values', 'dosing_times', 'dosing_routes', 'dosing_duration',
        'covariates', 'hidden_values', 'vector_field_at_hypercube', 'hypercube_locations',
        'model_parameters', 'error_parameters',
        'study_ids', 'hidden_process_dimension'
    ]
)

class CompartmentModel(abc.ABC):
    """
    Abstract class for the design of compartment models
    one must specify 

        -vector_field
        -get_observation_times
        -define_individual_params

    """
    name_str:str
    params:dataclass
    hidden_process_dimension_:int

    def __init__(self, params):
        """
        :param params: A dataclass containing model parameters.
        """
        self.params = params
        self.theta = self.define_individual_params()

    @abc.abstractmethod
    def vector_field(self, y, t, param)->np.array:
        pass

    @abc.abstractmethod
    def get_observation_times(self):
        pass

    @abc.abstractmethod
    def define_individual_params(self):
        pass

    @abc.abstractmethod
    def define_hypercube(self):
        pass

    def solve_ode(self,x0,t_obs,theta_i):
        """
        For  given parameters values solves one ode
        """
        return odeint(self.vector_field, 
                      x0, 
                      np.concatenate([[0], t_obs]), 
                      args=(theta_i,)).squeeze()

    def simulate(self)->FIMCompartementsDatabatchTuple:
        """
        First version using non efficient solver of odes
        for the compartment models

        (lorenz for example, is achieved in parallel using torch)
        """
        obs_values = []
        hidden_values = []
        obs_times = []
        dosing_values = []
        dosing_times = []
        vector_field_at_hypercube = []
        hypercube_locations = []
        study_ids = np.arange(self.params.N)
        hidden_process_dimension = 1  # Assumed to be 1 for simplicity

        for i in range(self.params.N):
            theta_i = self.theta[i, :]  # Individual's parameters
            x0 = [float(self.params.D)]  # Initial amount in plasma = dose
            dosing_time0 = [float(0.)]  # Time of dose

            dosing_values.append(np.array(x0))
            dosing_times.append(np.array(dosing_time0))

            # Solve ODE
            t_obs = self.get_observation_times()
            x_sol = self.solve_ode(x0, t_obs, theta_i)
            hidden_values.append(x_sol[1:])

            # Add residual variability
            noise = np.random.normal(0, self.params.sd_ruv, size=len(self.params.t_obs))
            obs_values.append((x_sol[1:] / theta_i[0]) + noise)
            obs_times.append(t_obs)

            # Placeholder for vector field and hypercube locations
            #vf_at_hypercube = self.vector_field(x_sol[1:], self.t_obs, theta_i)
            #vector_field_at_hypercube.append(vf_at_hypercube)
            #hypercube_locations.append(self.t_obs)  # Assuming obs times as hypercube locations

        dosing_values = np.stack(dosing_values, axis=0)
        hidden_values = np.stack(hidden_values, axis=0)
        obs_values = np.stack(obs_values, axis=0)
        obs_times = np.stack(obs_times, axis=0)

        # Placeholder arrays for other fields
        dosing_routes = [self.params.route] * self.params.N
        dosing_duration = [0.0] * self.params.N  # Assuming no duration for bolus
        covariates = np.zeros((self.params.N, 0))  # No covariates provided, set to empty
        model_parameters = self.theta  # Individual parameters
        error_parameters = np.array([self.params.sd_ruv])  # Error parameters

        #vector_field_at_hypercube = np.stack(vector_field_at_hypercube, axis=0)
        #hypercube_locations = np.stack(hypercube_locations, axis=0)
        return FIMCompartementsDatabatchTuple(
            obs_values=obs_values,
            obs_times=obs_times,
            dosing_values=dosing_values,
            dosing_times=dosing_times,
            dosing_routes=dosing_routes,
            dosing_duration=dosing_duration,
            covariates=None,
            hidden_values=hidden_values,
            vector_field_at_hypercube=None,
            hypercube_locations=None,
            model_parameters=model_parameters,
            error_parameters=error_parameters,
            study_ids=study_ids,
            hidden_process_dimension=hidden_process_dimension
        )

# ------------------------------------------------------------------------------------------
# Define the different compartment model parameters

@dataclass
class OneCompartmentModelParams:
    N: int = 1000  # Default number of individuals
    t_obs: np.ndarray = field(default_factory=lambda: np.array([0.5, 1, 2, 4, 8, 16, 24]))  # Default observation times
    D: float = 100  # Default dose administered (mg)
    route: str = 'iv_bolus'  # Default dosing route
    fe: dict = field(default_factory=lambda: {'V': 20, 'CL': 3})  # Default fixed effects (V, CL)
    sd_re: dict = field(default_factory=lambda: {'V': 0.4, 'CL': 0.3})  # Default random effects std devs (V, CL)
    sd_ruv: float = 0.2  # Default residual unexplained variability (mg/L)

# Example subclass for a one-compartment model
class OneCompartmentModel(CompartmentModel):

    name_str:str = "one_compartment"
    params:OneCompartmentModelParams = None
    hidden_process_dimension_:int = 1

    def vector_field(self, y, t, param):
        """
        One-compartment ODE system: dX/dt = -CL/V * X.
        :param y: Amount in the compartment.
        :param t: Time.
        :param param: Parameters [V, CL] (volume, clearance).
        :return: Rate of change of amount.
        """
        V, CL = param
        dX = - CL / V * y  # ODE for one-compartment model
        return dX

    def get_observation_times(self):
        return self.params.t_obs
    
    def define_individual_params(self):
        theta = np.zeros((self.params.N, 2))
        for i in range(self.params.N):
            theta[i, 0] = self.params.fe['V'] * np.random.lognormal(mean=0, sigma=self.params.sd_re['V'])
            theta[i, 1] = self.params.fe['CL'] * np.random.lognormal(mean=0, sigma=self.params.sd_re['CL'])
        return theta

    def define_hypercube(self):
        pass
    


