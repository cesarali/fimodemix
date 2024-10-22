import os
from pathlib import Path
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
import yaml
import numpy as np
from scipy.integrate import odeint
from collections import namedtuple
from scipy.integrate import odeint
from dataclasses import dataclass, field
from fimodemix.utils.grids import define_mesh_points

ROUTE_TO_IDS = {"iv_bolus":0,"iv_infusion":1,"oral":2}
STUDY_TO_IDS = {"one_compartment":0}


@dataclass
class CompartmentModelParams:
    compartment_name_str: str = "compartment" # USED TO REGISTER THE MODEL
    study_name_str:str = None # USED TO STORE THE MODEL SIMULATION
    redo_study:bool = False
    N: int = 1000  # Default number of individuals

class CompartmentModel(abc.ABC):
    """
    Abstract class for the design of compartment models
    one must specify 

        -vector_field
        -get_observation_times
        -define_individual_params
        -define_hypercube

        Global Variables:
            name_str (str): corresponds to the name of the particular
            compartment model and is used in the model registry for 
            later initialization, should match the name_str in the
            parameters dataclass

            params (dataclass): corresponds to the hyperparameters
            given so that one is able to simulate a given population
            study

            hidden_process_dimension_ (int): dimension of hidden process
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
    def f_vector_field(self, x, t, param)->np.array:
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

    def simulate(self)->FIMCompartementsDatabatch:
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
        covariates = np.zeros((self.params.N, 1))  # No covariates provided, set to empty
        # Individual parameters
        model_parameters = self.theta  
        # Error parameters
        error_parameters = np.full((self.params.N,1),self.params.sd_ruv)  

        data =  FIMCompartementsDatabatch(
            obs_values=obs_values,
            obs_times=obs_times,
            obs_mask=None,
            dosing_values=dosing_values,
            dosing_times=dosing_times,
            dosing_routes=dosing_routes,
            dosing_duration=dosing_duration,
            dosing_mask=None,
            covariates=covariates,
            hidden_values=hidden_values,
            vector_field_at_hypercube=vector_field_at_hypercube,
            hypercube_locations=hypercube_locations,
            model_parameters=model_parameters,
            error_parameters=error_parameters,
            study_ids=study_ids,
            hidden_process_dimension=hidden_process_dimension,
            dimension_mask=None
        )

        data.convert_to_tensors()
        return data

# ------------------------------------------------------------------------------------------
# Define the different compartment models

@dataclass
class OneCompartmentModelParams(CompartmentModelParams):
    compartment_name_str: str = "one_compartment"
    study_name_str:str = None
    N: int = 1000  # Default number of individuals
    t_obs: List[float] = field(default_factory=lambda: [0.5, 1, 2, 4, 8, 16, 24])  # Default observation times as a list
    D: float = 100.  # Default dose administered (mg)
    route: str = 'iv_bolus'  # Default dosing route
    fe: dict = field(default_factory=lambda: {'V': 20, 'CL': 3})  # Default fixed effects (V, CL)
    sd_re: dict = field(default_factory=lambda: {'V': 0.4, 'CL': 0.3})  # Default random effects std devs (V, CL)
    sd_ruv: float = 0.2  # Default residual unexplained variability (mg/L)
    num_hypercubes_points: int = 1024

# Example subclass for a one-compartment model
class OneCompartmentModel(CompartmentModel):

     #THIS IS THE KEY OF STUDY_TO_IDS
    name_str:str = "one_compartment"
    params:OneCompartmentModelParams = None
    hidden_process_dimension_:int = 1

    def __init__(self,params):
        super().__init__(params)
        self.params.t_obs = np.asarray(self.params.t_obs)

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

        dosing_routes.append(np.asarray([ROUTE_TO_IDS[self.params.route]]))
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
    
# ------------------------------------------------------------------------------------------
# MODEL REGISTRY

COMPARTMENT_NAMES_TO_MODELS = {
    "one_compartment":OneCompartmentModel
}

COMPARTMENT_NAMES_TO_PARMS = {
    "one_compartment":OneCompartmentModelParams
}

def set_up_a_study(
        params_yaml:dict,
        experiment_dir:str,
        return_data:bool=True,
    )->CompartmentModel|FIMCompartementsDatabatch:
    """
    Takes a dict of parameters from yaml and creates
    the Compartment model and generate the data accordingly
    every time the data is generated it will be saved

    Args:
        -params_yaml (dict): compartment model parameters as dict
        -experiment_dir (str): where all the models data is saved
        -return_data (bool): if true returns the FIMCompartementsDatabatch 
        otherwise the model
    
    Returns
        CompartmentModel|FIMCompartementsDatabatch
    """
    compartment_name_str = params_yaml["compartment_name_str"]
    study_name_str = params_yaml["study_name_str"]
    redo_study = params_yaml["redo_study"]
    study_path = Path(os.path.join(experiment_dir,study_name_str+".tr"))

    # Create an instance of OneCompartmentModelParams with the loaded values
    compartment_params = COMPARTMENT_NAMES_TO_PARMS[compartment_name_str](**params_yaml)
    compartment_model = COMPARTMENT_NAMES_TO_MODELS[compartment_name_str](compartment_params)

    if return_data:
        data:FIMCompartementsDatabatch
        # study data does not exist we generated again
        if not study_path.exists():
            data = compartment_model.simulate()
            torch.save(data,study_path)
            return data
        else:
            # data exist but we must simulate again
            if redo_study:
                data = compartment_model.simulate()
                torch.save(data,study_path)
                return data
            # data exist and we take it
            else:
                data = torch.load(study_path)
                return data
            
    return compartment_model

def define_compartment_models_from_yaml(
        yaml_file: str
    )->Tuple[str,
             List[CompartmentModel|FIMCompartementsDatabatch],
             List[CompartmentModel|FIMCompartementsDatabatch],
             List[CompartmentModel|FIMCompartementsDatabatch]]:
    """
    Function to load or generate different studies from a yaml file,
    this is the function that will allow the dataloader to get the data
    from the compartment studies
    
    Args:
        yaml_file: str of yaml file that contains a list of hyper parameters 
        from different compartment models, one such hyperparameters allows the 
        solver to generate one population study
    """
    from fimodemix import data_path
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    # check the experiment folder exist
    experiment_name = data["experiment_name"]
    experiment_dir = os.path.join(data_path,"compartment_model",experiment_name)
    if not os.path.exists(experiment_dir):
        # Create the folder
        os.makedirs(experiment_dir)

    # generate the data
    train_studies:List[CompartmentModel] = []
    test_studies:List[CompartmentModel] = []
    validation_studies:List[CompartmentModel] = []
    for params_yaml in data['train']:        
        compartment_model = set_up_a_study(params_yaml,experiment_dir)
        train_studies.append(compartment_model)

    for params_yaml in data['test']:        
        compartment_model = set_up_a_study(params_yaml,experiment_dir)
        test_studies.append(compartment_model)

    for params_yaml in data['validation']:        
        compartment_model = set_up_a_study(params_yaml,experiment_dir)
        validation_studies.append(compartment_model)

    return (
        experiment_name,
        train_studies,
        test_studies,
        validation_studies
    )