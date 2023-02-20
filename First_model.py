import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union

@dataclass
class Disease:
    beta: float
    gamma: float

class Compartment:
    def __init__(self, N: float = 10**6, businessmen_frac: float = 0.01, p_business: float = 0.5, stochastic: bool = True):
        """businessmen_frac is the fraction of the population that is in the businessmen class of people who travel all the time
        p_business is the fraction of people who travel who are from the business class
        """
        self.N_n = N * (1 - businessmen_frac)
        self.N_b = N * businessmen_frac
        self.businessmen_frac = businessmen_frac
        self.p_business = p_business
        self.stochastic = stochastic
        self.reset_parameters()
        # self.disease = disease
    
    def reset_parameters(self, I: Union[int,float] = 0):
        if self.stochastic:
            assert isinstance(I, int)
            self.I_n = [np.random.binomial(I, 1-self.businessmen_frac)]
        else:
            self.I_n = [I * (1-self.businessmen_frac)] 
        self.I_b = [I - self.I_n[0]]
        self.S_n = [self.N_n - self.I_n[0]]
        self.S_b = [self.N_b - self.I_b[0]]
        self.R = [0]
    
    def step(self, disease: Disease, timestep: float):
        #Mixing within communities
        

    




