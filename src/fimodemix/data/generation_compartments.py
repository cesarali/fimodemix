import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from fimodemix.utils.grids import define_mesh_points

from typing import List,Tuple,Union,Optional
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

from fimodemix.utils.grids import define_mesh_points

ROUTE_TO_IDS = {"iv_bolus":0,"iv_infusion":1,"oral":2}
STUDY_TO_IDS = {"one_compartment":0}

# Define the namedtuple
FIMCompartementsDatabatchTuple = namedtuple(
    'FIMCompartementsDatabatchTuple',
    [
        'obs_values', 'obs_times','obs_mask',
        'dosing_values', 'dosing_times', 'dosing_routes', 'dosing_duration','dosing_mask',
        'covariates', 'hidden_values', 'vector_field_at_hypercube', 'hypercube_locations',
        'model_parameters', 'error_parameters',
        'study_ids', 'hidden_process_dimension','dimension_mask'
    ]
)

class CompartmentModel(abc.ABC):
    """
    Abstract class for the design of compartment models
    one must specify 

        -vector_field
        -get_observation_times
        -define_individual_params
        -define_hypercube

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
    def f_vector_field(self, y, t, param)->np.array:
        pass
    
    @abc.abstractmethod
    def get_observation_times(self):
        pass
    
    @abc.abstractmethod
    def get_dosing(self,dosing_values,dosing_times,dosing_routes,dosing_duration)->Tuple[np.array,List,List,List,List]:
        pass

    @abc.abstractmethod
    def define_individual_params(self)->np.array:
        pass

    @abc.abstractmethod
    def define_hypercube(self)->np.array:
        pass

    def solve_ode(self,x0,t_obs,theta_i)->np.array:
        """
        For  given parameters values solves one ode
        """
        return odeint(self.f_vector_field, 
                      x0, 
                      np.concatenate([[0], t_obs]), 
                      args=(theta_i,)).squeeze()

    def simulate(self)->FIMCompartementsDatabatchTuple:
        """
        First version using non efficient solver of odes
        for the compartment models.

        THIS WILL BE ADAPTED TO ACCOUNT FOR MORE 
        COMPLEX DOSING

        (lorenz for example, is achieved in parallel using torch)
        """
        obs_values = []
        hidden_values = []
        obs_times = []

        dosing_values = []
        dosing_times = []
        dosing_routes = []
        dosing_duration = []

        vector_field_at_hypercube = []
        hypercube_locations = []

        # Study Ids
        study_ids = np.full((self.params.N,1),STUDY_TO_IDS[self.name_str])
        # Assumed to be 1 for simplicity
        hidden_process_dimension = np.full((self.params.N,1),self.hidden_process_dimension_)
        
        for i in range(self.params.N):
            # Individual's parameters
            theta_i = self.theta[i, :]
  
            # Get dosing
            x0,dosing_values,dosing_times,dosing_routes,dosing_duration = self.get_dosing(
                dosing_values,
                dosing_times,
                dosing_routes,
                dosing_duration
                )

            # Solve ODE
            t_obs = self.get_observation_times()
            x_sol = self.solve_ode(x0, t_obs, theta_i)
            hidden_values.append(x_sol[1:])

            # Add residual variability
            noise = np.random.normal(0, self.params.sd_ruv, size=len(self.params.t_obs))
            obs_values.append((x_sol[1:] / theta_i[0]) + noise)
            obs_times.append(t_obs)

            # Placeholder for vector field and hypercube locations
            hypercube = self.define_hypercube()
            vf_at_hypercube = self.f_vector_field(hypercube, t_obs, theta_i)
            vector_field_at_hypercube.append(vf_at_hypercube)
            hypercube_locations.append(hypercube)  # Assuming obs times as hypercube locations

        # Convert lists to torch tensors
        dosing_values = np.stack(dosing_values,axis=0)
        dosing_times = np.stack(dosing_times,axis=0)
        dosing_duration = np.stack(dosing_duration,axis=0)
        dosing_routes = np.stack(dosing_routes,axis=0)
        hidden_values = np.stack(hidden_values,axis=0)
        obs_values = np.stack(obs_values,axis=0)
        obs_times = np.stack(obs_times,axis=0)

        vector_field_at_hypercube = np.stack(vector_field_at_hypercube,axis=0)
        hypercube_locations = np.stack(hypercube_locations,axis=0)

        # Placeholder arrays for other fields
        covariates = np.zeros((self.params.N, 0))  # No covariates provided, set to empty

        # Individual parameters
        model_parameters = self.theta  
        # Error parameters
        error_parameters = np.full((self.params.N,1),self.params.sd_ruv)  


        return FIMCompartementsDatabatchTuple(
            obs_values=obs_values,
            obs_times=obs_times,
            obs_mask=None,
            dosing_values=dosing_values,
            dosing_times=dosing_times,
            dosing_routes=dosing_routes,
            dosing_duration=dosing_duration,
            dosing_mask=None,
            covariates=None,
            hidden_values=hidden_values,
            vector_field_at_hypercube=vector_field_at_hypercube,
            hypercube_locations=hypercube_locations,
            model_parameters=model_parameters,
            error_parameters=error_parameters,
            study_ids=study_ids,
            hidden_process_dimension=hidden_process_dimension,
            dimension_mask=None
        )

# ------------------------------------------------------------------------------------------
# Define the different compartment model parameters

@dataclass
class OneCompartmentModelParams:
    N: int = 1000  # Default number of individuals
    t_obs: np.ndarray = field(default_factory=lambda: np.array([0.5, 1, 2, 4, 8, 16, 24]))  # Default observation times
    D: float = 100.  # Default dose administered (mg)
    route: str = 'iv_bolus'  # Default dosing route
    fe: dict = field(default_factory=lambda: {'V': 20, 'CL': 3})  # Default fixed effects (V, CL)
    sd_re: dict = field(default_factory=lambda: {'V': 0.4, 'CL': 0.3})  # Default random effects std devs (V, CL)
    sd_ruv: float = 0.2  # Default residual unexplained variability (mg/L)

    num_hypercubes_points:int = 1024

# Example subclass for a one-compartment model
class OneCompartmentModel(CompartmentModel):

    name_str:str = "one_compartment" #THIS IS THE KEY OF STUDY_TO_IDS
    params:OneCompartmentModelParams = None
    hidden_process_dimension_:int = 1

    def f_vector_field(
            self, 
            y:np.array, 
            t:Optional[np.array], 
            param:np.array
        )->np.array:
        """
        One-compartment ODE system: dX/dt = -CL/V * X.

        :param y: Amount in the compartment.
        :param t: Time.
        :param param: Parameters [V, CL] (volume, clearance).
        :return: Rate of change of amount.
        """
        if len(y.shape) == 1:
            V, CL = param
            dX = - CL / V * y  # ODE for one-compartment model
            return dX
        else:
            dX = - param[None,1] / param[None,0] * y  # ODE for one-compartment model
            return dX

    def get_observation_times(self)->np.array:
        """for this models we have the same observation times for every patient"""
        return self.params.t_obs

    def get_dosing(
            self,
            dosing_values,
            dosing_times,
            dosing_routes,
            dosing_duration
        )->Tuple[np.array,List,List,List,List]:
        """simple models assumes initial condition as the dosing"""
        x0 = [float(self.params.D)]  # Initial amount in plasma = dose
        dosing_time0 = [float(0.)]  # Time of dose
        dosing_values.append(np.array(x0))
        dosing_times.append(np.array(dosing_time0))

        dosing_routes.append(np.asarray([self.params.route]))
        dosing_duration.append(np.array([0.]))  # Assuming no duration for bolus

        return x0,dosing_values,dosing_times,dosing_routes,dosing_duration

    def define_individual_params(self)->np.array:
        """individual parameters per patient"""
        theta = np.zeros((self.params.N, 2))
        for i in range(self.params.N):
            theta[i, 0] = self.params.fe['V'] * np.random.lognormal(mean=0, sigma=self.params.sd_re['V'])
            theta[i, 1] = self.params.fe['CL'] * np.random.lognormal(mean=0, sigma=self.params.sd_re['CL'])
        return theta

    def define_hypercube(self)->np.array:
        hypercube_locations = define_mesh_points(self.params.num_hypercubes_points,
                                                 n_dims=1,
                                                 ranges=[0.,self.params.D]).detach().numpy()
        return hypercube_locations
    

