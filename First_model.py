import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union

@dataclass
class Disease:
    beta: float
    gamma: float
    delta: float

class Compartment:
    def __init__(self, N: int = 10**6, businessmen_frac: float = 0.01, p_business: float = 0.5, business_mixing:float = 0.5, stochastic: bool = True):
        """businessmen_frac is the fraction of the population that is in the businessmen class of people who travel all the time
        p_business is the fraction of people who travel who are from the business class
        business_mixing is a parameter which detemines how much businessmen interact with the rest of society: the effective beta for interactions between business and non business people is beta * business_integration
        """
        self.N = {'normal': int(N * (1 - businessmen_frac)),
                  'business': int(N * businessmen_frac)}
        self.businessmen_frac = businessmen_frac
        self.p_business = p_business
        self.business_mixing = business_mixing
        self.stochastic = stochastic
        self.reset_parameters()
        # self.disease = disease
    
    def reset_parameters(self, I: Union[int,float] = 0):
        if self.stochastic:
            assert isinstance(I, int)
            I_n = [np.random.binomial(I, 1-self.businessmen_frac)]
        else:
            I_n = [I * (1-self.businessmen_frac)] 
        I_b = [I - I_n[0]]
        self.I = {'normal': I_n, 'business': I_b}
        self.S = {'normal': [self.N['normal'] - I_n[0]], 'business': [self.N['business'] - I_b[0]]}
        self.E = {'normal': [0], 'business': [0]}
        self.R = {'normal': [0], 'business': [0]}
    
    def simulate(self, time_interval:float, epidemic_time:float, disease:Disease):
        p_recovery = 1 - np.exp( - time_interval * disease.gamma)
        p_infectious = 1 - np.exp( - time_interval * disease.delta)
        timesteps = int(epidemic_time // time_interval)
        self.times = np.arange(0,epidemic_time,time_interval)
        for _ in range(timesteps):
            self.internal_spread(disease.beta,
                      disease.gamma,
                      time_interval,
                      p_infectious,
                      p_recovery)

    def internal_spread(self,
                        beta:float,
                        gamma:float,
                        time_interval: float,
                        p_infectious,
                        p_recovery):
        new_S = {}
        new_E = {}
        new_I = {}
        new_R = {}

        #Mixing between non_businessmen and businessmen separately
        for group in ['normal', 'business']:
            exposure_rate = beta * self.I[group][-1] / self.N[group]
            p_exposure = 1 - np.exp(- time_interval * exposure_rate)
            n_exposed = np.random.binomial(self.S[group][-1], p_exposure)
            n_infectious = np.random.binomial(self.E[group][-1], p_infectious)
            n_recovered = np.random.binomial(self.I[group][-1], p_recovery)
            new_S[group] = self.S[group][-1] - n_exposed
            new_E[group] = self.E[group][-1] + n_exposed - n_infectious
            new_I[group] = self.I[group][-1] + n_infectious - n_recovered
            new_R[group] = self.R[group][-1] + n_recovered
        
        #Mixing between businessmen and nonbusinessmen
        groups = ['normal', 'business']
        for i, group in enumerate(groups):
            exposure_rate = beta * new_I[groups[not i]] / (self.N[groups[0]] + self.N[groups[1]])
            p_exposure = 1 - np.exp(- time_interval * exposure_rate)
            n_exposed = np.random.binomial(new_S[group], p_exposure)
            new_S[group] -= n_exposed
            new_E[group] += n_exposed

            self.S[group].append(new_S[group])
            self.E[group].append(new_E[group])
            self.I[group].append(new_I[group])
            self.R[group].append(new_R[group])
        
    def plot_compartment(self):
        fig, axs = plt.subplots(1,2)
        groups = ['normal', 'business']
        for i, group in enumerate(groups):
            axs[i].plot(self.times, self.S[group], label = 'S')
            axs[i].plot(self.times, self.E[group], label = 'E')
            axs[i].plot(self.times, self.I[group], label = 'I')
            axs[i].plot(self.times, self.R[group], label = 'R')
            axs[i].legend()
            axs[i].set_title(group)
        print('hello')
        fig.show()

measles = Disease(1.5, 1/8, 1/10)
test = Compartment()
test.simulate(0.2,50,measles)     
test.plot_compartment()

        

plt.plot(list(range(10)), list(range(10)))