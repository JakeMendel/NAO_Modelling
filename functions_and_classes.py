#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dataclasses import dataclass, field
from typing import Union, Tuple, Iterable, Optional, Any, Type, Callable, Dict
from tqdm import tqdm
from fancy_einsum import einsum

#%%
class Disease:
    def __init__(self, beta: float, gamma: float, delta: float):
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.doubling_time, self.growth_rate, self.decay_rate = self.effective_growth_rate()

    def effective_growth_rate(self):
        '''
        Returns the effective growth rate for the epidemic in the exponential phase
        '''
        b = self.beta
        g = self.gamma
        d = self.delta
        growth_rate = 0.5 * (np.sqrt((d-g)**2 +4*d*b) - d - g)
        doubling_time = np.log(2)/growth_rate
        decay_rate = -0.5 * (d + g + np.sqrt((d-g)**2 +4*d*b))
        return doubling_time, growth_rate, decay_rate
    
    def __str__(self):
        return str(vars(self))
#%%
covid = Disease(0.6,1/6,1/5.5)
measles = Disease(2,1/10,1/8)
#%%
def find_first_index(condition, axis = -1):
    """
    Finds the index of the first time a condition is met for each row in a 2D numpy array.

    Args:
        array: A 2D numpy array with n rows and m columns.
        condition: A boolean condition to be checked element-wise.

    Returns:
        A 1D numpy array containing the index of the first time the condition is met for each row in the input array.
        Returns -1 if the condition is never met in a row.
    """
    assert isinstance(condition, np.ndarray), "array must be a numpy array"
    assert len(condition.shape) == 2, "array must be a 2D numpy array"
    assert condition.shape[1] > 0, "array must have at least one column"

    # Apply the condition element-wise to the input array and get the index of the first True value along axis 1
    idx = np.argmax(condition, axis=axis)
    
    # If the condition is never met in a row, set the corresponding index to -1
    idx[np.all(~condition, axis=axis)] = -1
    
    return idx

def homogeneous_LR_matrix(n_groups: int, LR: float):
    out = np.identity(n_groups,np.float64)
    out[out == 0] = LR
    return out

def day_to_sim_step(day, delta_t):
    #day should start at zero
    steps_in_day = int(1/delta_t)
    return np.round(steps_in_day * (day + 0.5)).astype(int)

def sim_step_to_day(sim_step, delta_t):
    return np.floor(sim_step * delta_t).astype(int)

def quartiles(data: np.ndarray, axis: int = -1, squeeze = True):
    'returns lower_quartile, median, upper_quartile'
    return tuple(np.quantile(data, [0.25,0.5,0.75],axis=axis).squeeze())

def list_to_str(l: Iterable, sep: str = ''):
    out = ''
    for i in l:
        out+= str(i) + sep
    return out[:-len(sep)]

def dict_to_str(d: dict, sep: str = ', '):
    out = ''
    for key, val in d.items():
        assert isinstance(val, list) or isinstance(val, str)
        val_str = list_to_str(val,sep) if isinstance(val, list) else val
        out += f'{key}: {val_str}{sep}'
    return out[:-len(sep)]

def force_iterable(input):
    return input if isinstance(input, Iterable) else [input]

def force_list(input):
    return list(input) if isinstance(input, Iterable) and not isinstance(input,str) else [input]

def show_fig(fig, figsavename):
    if figsavename is not None:
        fig.savefig(figsavename)
    else:
        fig.show()

def total_daily(array: np.ndarray, delta_t: float, axis: int = -1):
    #the timeseries axis must be the last axis!
    assert axis == -1, 'the timeseries axis must be the last axis!'
    window = int(1/delta_t)
    shape = array.shape
    assert (len(shape) == 2) or (len(shape) == 1)
    assert shape[-1] % window == 0
    return array.reshape(shape[:-1] + (shape[-1]//window,window)).sum(axis=-1)

def mean_daily(array: np.ndarray, delta_t: float, axis: int = -1):
    #the timeseries axis must be the last axis!
    assert axis == -1, 'the timeseries axis must be the last axis!'
    window = int(1/delta_t)
    shape = array.shape
    assert (len(shape) == 2) or (len(shape) == 1)
    assert shape[-1] % window == 0
    return array.reshape(shape[:-1] + (shape[-1]//window,window)).mean(axis=-1)
     
#%%
class City:
    def __init__(self,
                 N0s: np.ndarray,
                 groups: list[str],
                 compartments: str = 'SEIR',
                 group_LR: Optional[np.ndarray] = None):
        """
        Initializes the City object.

        Args:
        - N0s (np.ndarray): An array of initial population sizes for each group.
        - groups (list): A list representing the groups in the model.
        - compartments (str): A string representing the compartment model used in the model. Currently only 'SEIR' is supported.
        - group_LR (np.ndarray): An optional array representing the matrix of inter-group interaction likelihood ratios for groups i and j. All diagonal elements should be one.

        Returns:
        - None
        """
        self.n_groups = len(groups)
        self.groups = groups

        # Check that the length of N0s matches the number of groups
        assert len(N0s) == self.n_groups, "N0s should have length n_groups"

        self.N0s = N0s
        self.N0 = self.N0s.sum()

        assert compartments == 'SEIR', "Only SEIR compartment model supported at the moment"
        self.model_type = compartments
        self.compartments = [l for l in compartments]

        # Check and set the group_LR matrix
        if group_LR is None:
            group_LR = np.identity(self.n_groups)
        assert group_LR.shape == (self.n_groups, self.n_groups)
        assert np.all(group_LR.diagonal() == 1)
        assert np.all(group_LR == group_LR.T)
        self.group_LR = group_LR

        # Set the indices of the compartments
        self.S_index = self.compartments.index('S')
        self.E_index = self.compartments.index('E')
        self.I_index = self.compartments.index('I')
        self.R_index = self.compartments.index('R')

        self.name: Optional[Union[int,str]] = None
        self.disease: Optional[Disease] = None

        self.data_cols = {'municipal': 'blue', 'arrivals': 'red', 'departures': 'green'}
        self.error_bar_cols = {'municipal': 'cornflowerblue', 'arrivals': 'lightcoral', 'departures': 'mediumaquamarine'}

    def reset_parameters(self, I0: int = 0, n_sims: int = 1, simulation_steps: int = 100):
        """
        Sets self.municipal to an array of zeros of size (n_groups, n_compartments, n_sims, timesteps) for each compartment in the city
        The only values which aren't zero are the initial value of I which is I0 and the initial value of S which is N-I0
        The default value of I0 is the value set in the class initialization, but can be overridden
        """
        # if I0 is None:
        #     I0 = self.I0
        self.I0 = I0
        self.municipal = np.zeros((self.n_groups,len(self.compartments),self.n_sims, self.simulation_steps),dtype = np.int64)
        self.arrivals = np.zeros((self.n_groups,len(self.compartments),self.n_sims, self.simulation_steps),dtype = np.int64)
        self.departures = np.zeros((self.n_groups,len(self.compartments),self.n_sims, self.simulation_steps),dtype = np.int64)
        self.community_infections = np.zeros((self.n_groups, self.n_sims, self.simulation_steps), dtype=np.int64)
        self.initial_conditions()
    
    def initial_conditions(self):
        raise ModuleNotFoundError
    
    def multiple_sims(self, delta_t: float, epidemic_time: Union[int,float], disease: Disease, I0: int = 0, n_sims: int = 100):
        assert (1 / delta_t) % 1 == 0, "1/delta_t must be an integer"
        self.n_sims = n_sims
        self.delta_t = delta_t
        self.epidemic_time = epidemic_time
        p_recovery = 1 - np.exp( - delta_t * disease.gamma)
        p_infectious = 1 - np.exp( - delta_t * disease.delta)
        self.simulation_steps = int(epidemic_time // delta_t) + 1
        self.times: np.ndarray = np.linspace(0, epidemic_time, self.simulation_steps)
        self.scaled_times: np.ndarray = self.times /disease.doubling_time
        self.reset_parameters(I0, n_sims, self.simulation_steps)
        for sim_step in tqdm(range(1,self.simulation_steps)):
            self.step_internal(disease.beta,
                      delta_t,
                      p_infectious,
                      p_recovery,
                      sim_step)
        self.daily_flight_data()
    
    def __call__(self,
                 delta_t: float,
                 epidemic_time: Union[int,float],
                 disease: Disease,
                 I0: int = 0,
                 n_sims: int = 100):
        return self.multiple_sims(delta_t, epidemic_time, disease, I0, n_sims)

    def step_internal(self,
                      beta:float,
                      delta_t: float,
                      p_infectious: float,
                      p_recovery: float,
                      simulation_step: int):
        N = self.municipal[...,simulation_step-1].sum(axis = 1)
        
        S = self.municipal[:,self.S_index,:,simulation_step-1]
        E = self.municipal[:,self.E_index,:,simulation_step-1]
        I = self.municipal[:,self.I_index,:,simulation_step-1]
        R = self.municipal[:,self.R_index,:,simulation_step-1]


        modified_I = einsum('group1 group2, group2 n_sims -> group1 n_sims', self.group_LR, I)
        modified_N = einsum('group1 group2, group2 n_sims -> group1 n_sims', self.group_LR, N)
        exposure_rate = beta * modified_I/modified_N
        p_exposure = 1 - np.exp(- delta_t * exposure_rate)
        n_exposed = np.random.binomial(S, p_exposure)
        n_infectious = np.random.binomial(E,p_infectious)
        n_recovered = np.random.binomial(I,p_recovery)

        self.municipal[:,self.S_index,:,simulation_step] = S - n_exposed
        self.municipal[:,self.E_index,:,simulation_step] = E + n_exposed - n_infectious
        self.municipal[:,self.I_index,:,simulation_step] = I + n_infectious - n_recovered
        self.municipal[:,self.R_index,:,simulation_step] = R + n_recovered
        self.community_infections[..., simulation_step] = self.community_infections[..., simulation_step-1] + n_exposed

    def daily_flight_data(self, moving_avg = False):
        window = int(1/self.delta_t)
        
        if moving_avg:
            kernel = np.zeros(2*self.simulation_steps-1)
            kernel[self.simulation_steps-1:self.simulation_steps+window-1] = 1
            strided_kernel = np.flip(np.lib.stride_tricks.sliding_window_view(kernel, self.simulation_steps).astype(int).copy(),axis=0)
            outputs = [np.zeros_like(self.arrivals)] * 2
            for i,data in tqdm(enumerate([self.arrivals, self.departures])):
                output = einsum('groups compartments sims simsteps, kernelsteps simsteps -> groups compartments sims kernelsteps', data, strided_kernel)
                # if i == 0:
                #     print('data', data)
                #     print('kernel', strided_kernel)
                #     print('output', output)
                outputs[i] = output
            self.arrivals_moving_avg, self.departures_moving_avg = outputs[0], outputs[1]
            
        self.arrivals_daily_avg = self.arrivals.reshape((self.n_groups, len(self.compartments), self.n_sims, self.simulation_steps//window, window)).sum(axis = -1)
        self.departures_daily_avg = self.departures.reshape((self.n_groups, len(self.compartments), self.n_sims, self.simulation_steps//window, window)).sum(axis = -1)

    def select_travellers(self, daily_mixnumber: int, simulation_step: int):
        raise ModuleNotFoundError

    def plot_sims(self,
                  times: Optional[np.ndarray] = None,
                  cityname: Union[int,str] = 0, 
                  shift_index: Optional[np.ndarray] = None,
                  separate_groups: bool = False,
                  figsavename: Optional[str] = None,
                  moving_avg = False,
                  log = ''):
        include_flight_data = 'arrivals' in dir(self)
        subplots = self.n_groups if separate_groups else 1
        if include_flight_data:
            fig, axs = plt.subplots(3,subplots, figsize = (20,30))
            axs = np.array(axs)
        else:
            fig, axs = plt.subplots(1,subplots, figsize = (20,10))
            axs = np.expand_dims(np.array(axs),0)
        
        for ax in axs:
            if 'x' in log:
                ax.set_xscale('log')
            if 'y' in log:
                ax.set_yscale('log')
        
        if self.n_groups == 1:
            axs = np.expand_dims(axs, -1)
        if times is None:
            times = self.times
        if shift_index is None:
            shift = np.zeros((self.n_sims,1))
        else:
            shift = times[shift_index].reshape((self.n_sims,1))
        days = np.array(range(int(max(times))), dtype = np.float64)

        times = times - shift + shift.mean()
        days = days - shift + shift.mean()
        
        travel_times = times if moving_avg else days
        arrivals = self.arrivals_moving_avg if moving_avg else self.arrivals_daily_avg
        departures = self.departures_moving_avg if moving_avg else self.departures_daily_avg

        cols = ['green', 'orange', 'red', 'blue']
        labels = ['Municipal', 'Arrivals', 'Departures']
        if separate_groups:
            for i, group in enumerate(self.groups):
                for j,compartment in tqdm(enumerate(self.compartments)):
                    axs[0,i].plot(times[0],
                                self.municipal[i,j,0],
                                label = compartment,
                                color = cols[j])
                    if include_flight_data:
                        axs[1,i].plot(travel_times[0],  
                                    arrivals[i,j,0],
                                    label = compartment,
                                    color = cols[j])
                        axs[2,i].plot(travel_times[0],
                                    departures[i,j,0],
                                    label = compartment,
                                    color = cols[j])
                    for k,datum in enumerate(self.municipal[i,j,1:]):
                        axs[0,i].plot(times[k+1], datum, color = cols[j])
                    if include_flight_data:
                        for k, (arrival, departure) in enumerate(zip(arrivals[i,j,1:],departures[i,j,1:])):
                            axs[1,i].plot(travel_times[k+1], arrival, color = cols[j])
                            axs[2,i].plot(travel_times[k+1], departure, color = cols[j])
                num_subplots = axs.shape[0]
                for j in range(num_subplots):
                    axs[j,i].legend()
                    axs[j,i].set_title(f"City {cityname}, {group}: {labels[j]}")
        else:
            for j, compartment in tqdm(enumerate(self.compartments)):
                axs[0].plot(times[0],
                            self.municipal[:,j,0].sum(axis = 0),
                            label = compartment,
                            color = cols[j])
                if include_flight_data:
                    axs[1].plot(travel_times[0],
                                arrivals[:,j,0].sum(axis = 0),
                                label = compartment,
                                color = cols[j])
                    axs[2].plot(travel_times[0],
                                departures[:,j,0].sum(axis = 0),
                                label = compartment,
                                color = cols[j])
                for k,datum in enumerate(self.municipal[:,j,1:].sum(axis = 0)):
                    axs[0].plot(times[k+1], datum, color = cols[j])
                    if include_flight_data:
                        for k, (arrival, departure) in enumerate(zip(arrivals[:,j,1:].sum(axis = 0),departures[:,j,1:].sum(axis = 0))):
                            axs[1].plot(travel_times[k+1], arrival, color = cols[j])
                            axs[2].plot(travel_times[k+1], departure, color = cols[j])
            num_subplots = axs.shape[0]
            for j in range(num_subplots):
                axs[j].legend()
                axs[j].set_title(f"City {cityname}: {labels[j]}")
        show_fig(fig,figsavename)
    
    def peak_I_times(self):
        return self.municipal[:,self.I_index].sum(axis = 0).argmax(axis = -1)

    def __str__(self):
        raise ModuleNotFoundError
    
    def sim_steps_from_times(self,times: Union[float,np.ndarray]):
        return np.argmin(np.abs(self.times-np.expand_dims(times,-1)),axis=-1)
    
    def times_from_sim_steps(self,sim_steps: Union[int,np.ndarray]):
        return sim_steps * self.delta_t

#%%
class FrequentFlyerCity(City):
    def __init__(self,
                 N0: int = 10**6,
                 frequent_flyer_frac: float = 0.1,
                 p_ff: Optional[float] = None,
                 flying_LR: Optional[float] = None,
                 group_LR: float = 5,
                 compartments: str = 'SEIR'):
        group_LR_matrix = homogeneous_LR_matrix(2,1/group_LR)
        self.groups = ['normal', 'frequent_flyers']
        self.N0s = np.array([N0 * (1 - frequent_flyer_frac), N0 * frequent_flyer_frac],dtype = np.int64)
        super().__init__(self.N0s, self.groups, compartments, group_LR_matrix)
        self.frequent_flyer_frac = frequent_flyer_frac
        if (p_ff is None) and (flying_LR is None):
            flying_LR = 10
        assert (p_ff is None) != (flying_LR is None), "Specify exactly one of p_ff OR flying_LR!"
        if flying_LR is not None:
            self.flying_LR = flying_LR
            self.p_ff = flying_LR * frequent_flyer_frac/ (flying_LR * frequent_flyer_frac + (1 - frequent_flyer_frac))
        if p_ff is not None:
            self.p_ff = p_ff
            self.flying_LR = (p_ff / frequent_flyer_frac) / ((1-p_ff) / (1 - frequent_flyer_frac))

    def initial_conditions(self):
        I_n = np.random.binomial(self.I0, 1-self.frequent_flyer_frac, self.n_sims)
        I_ff = self.I0 - I_n
        self.municipal[0,2,:,0] = I_n
        self.municipal[1,2,:,0] = I_ff
        self.municipal[0,0,:,0] = self.N0s[0] - I_n
        self.municipal[1,0,:,0] = self.N0s[1] - I_ff
    
    def select_travellers(self, daily_mixnumber: int, simulation_step: int):
        "Selects travellers based on a constant rate per person. This means number of travellers isn't preserved so the rate is modified to provide a force that pushes things back to the baseline population"
        # N = []
        fractions = np.array([1 - self.p_ff, self.p_ff])
        p_travel = 1 - np.exp( - self.delta_t * daily_mixnumber * fractions / self.N0s)
        return np.random.binomial(self.municipal[...,simulation_step], p_travel.reshape((2,1,1)))

    def __str__(self):
        out = 'City Type:\n FrequentflyerCity'
        out += f'\nN:\n {self.N0}'
        out += f'\nfrequent_flyer_frac:\n {self.frequent_flyer_frac}'
        out += f'\nflying_LR:\n {self.flying_LR}'
        out += f'\np_ff:\n {self.p_ff}'
        out += f'\ngroup_LR:\n {self.group_LR}'
        return out


#%%
class BasicCity(City):
    def __init__(self,
                 N0: int = 10**6,
                 compartments: str = 'SEIR'):
        group_LR_matrix = np.array([[1.]])
        self.groups = ['normal']
        self.N0s = np.array([N0],dtype = np.int64)
        super().__init__(self.N0s, self.groups, compartments, group_LR_matrix)
    
    def initial_conditions(self):
        self.municipal[0,2,:,0] = self.I0
        self.municipal[0,0,:,0] = self.N0s[0] - self.I0
    
    def __str__(self):
        out = 'City Type:\n BasicCity'
        out += f'\nN:\n {self.N0}'
        return out

    def select_travellers(self, daily_mixnumber: int, simulation_step: int):
        p_travel = 1 - np.exp(- self.delta_t * daily_mixnumber / self.N0)
        return np.random.binomial(self.municipal[...,simulation_step],p_travel)

#%%
def default_axis_order():
    return ['cities', 'datatypes', 'groups', 'compartments', 'sims', 'times']

@dataclass
class SimData:
    array: np.ndarray
    values_present: Dict
    axis_order: list[str] = field(default_factory=default_axis_order)

    def __getitem__(self,to_keep):
        assert isinstance(self.values_present,dict)
        values_to_keep = self.values_present.copy()
        if isinstance(to_keep, list) or isinstance(to_keep, tuple):
            for category, elemtype in zip(to_keep, values_to_keep.keys()):
                values_to_keep[elemtype] = category
        elif isinstance(to_keep, dict):
            for key, value in to_keep.items():
                values_to_keep[key] = value

        new_all_values = {}
        for key,value in self.values_present.items():
            new_all_values[key] = []
            for elem in value:
                if elem in values_to_keep[key]:
                    new_all_values[key].append(elem)

        chosen_indices = {}
        for key in self.values_present.keys():
            chosen_indices[key] = np.zeros(len(self.values_present[key])).astype(bool)
            for i,value in enumerate(self.values_present[key]):
                if value in values_to_keep[key]:
                    chosen_indices[key][i] = True

            for value in values_to_keep[key]:
                assert value in self.values_present[key], f'{value} is not a member of {key}, which currently only contains {self.values_present[key]}'
        out_array = self.choose_array(chosen_indices)
        return SimData(out_array, values_present = new_all_values, axis_order=self.axis_order)
    
    def __eq__(self,other):
        array_bool = np.all(self.array - other.array == 0)
        label_bool = self.values_present == other.all_labels
        return array_bool and label_bool
    
    def filter(self, to_keep):
        return self.__getitem__(to_keep)
    
    def choose_array(self, chosen_indices: dict):
        out_array = self.array
        for key, value in chosen_indices.items():
            axis = self.axis_order.index(key)
            out_array = np.compress(value, out_array, axis = axis)
        return out_array
    
    def wrap_np_function(self,
                         np_function: Callable,
                         axis: Union[int, str, Iterable] = 0,
                         keepdims: bool = False,
                         SimData_out = True,
                         **kwargs):
        axislist = force_list(axis)
        axisnums = [self.axis_order.index(ax) if isinstance(ax,str) else ax for ax in axislist]

        if not SimData_out:
            return np_function(self.array, axis = tuple(axisnums), keepdims = keepdims, **kwargs)
        
        axistypes = [ax if isinstance(ax,str) else self.axis_order[ax] for ax in axislist]
        new_axes = self.axis_order if keepdims else [ax for ax in self.axis_order if ax not in axistypes]
        new_labels = self.values_present.copy()
        for ax in axistypes:
            if not keepdims:
                del new_labels[ax]
            else:
                new_labels[ax] = ['N/A']
        new_arr = np_function(self.array, axis = tuple(axisnums), keepdims = keepdims, **kwargs)
        return SimData(new_arr, new_labels, new_axes)
    
    def wrap_np_binary_operator(self, np_function: Callable, other, SimData_out: bool = True):
        if isinstance(other, SimData):
            assert self.axis_order == other.axis_order
            assert self.values_present == other.values_present
            other_arr = other.array
        else:
            other_arr = other
        out_arr = np_function(self.array, other_arr)
        return SimData(out_arr,self.values_present, self.axis_order) if SimData_out else out_arr

    def sum(self, axis: Union[int, str, Iterable] = 0, keepdims: bool = False, SimData_out: bool = True):
        return self.wrap_np_function(np.sum,axis,keepdims, SimData_out)
    
    def mean(self, axis: Union[int, str, Iterable] = 0, keepdims: bool = False, SimData_out: bool = True):
        return self.wrap_np_function(np.mean,axis,keepdims, SimData_out)
    
    def std(self, axis: Union[int, str, Iterable] = 0, keepdims: bool = False, SimData_out: bool = True):
        return self.wrap_np_function(np.std,axis,keepdims, SimData_out)
    
    def quartiles(self, axis: Union[int, str, Iterable] = 0, keepdims: bool = False, SimData_out: bool = True):
        out = []
        for i,quartile in enumerate([0.25,0.5,0.75]):
            out.append(self.wrap_np_function(np.quantile,axis,keepdims, SimData_out, q=quartile))
        return tuple(out)
    
    def __add__(self, other):
        return self.wrap_np_binary_operator(np.add, other)
    
    def __truediv__(self, other):
        return self.wrap_np_binary_operator(np.divide, other)
    
    def max(self, axis: Union[int, str, Iterable] = 0, keepdims: bool = False, SimData_out: bool = True):
        return self.wrap_np_function(np.max,axis,keepdims, SimData_out)
    
    def argmax(self, axis: Union[int, str, Iterable] = 0, keepdims: bool = False, SimData_out: bool = True):
        return self.wrap_np_function(np.argmax, axis, keepdims, SimData_out)
    
    def daily_avg(self):
        old_times = self.values_present['times']
        delta_t = old_times[1] - old_times[0]
        window = round(1/delta_t)
        
        new_shape = list(self.array.shape)
        new_shape[self.axis_order.index('times')] //= window
        new_shape = tuple(new_shape+[window])

        new_arr = self.array.reshape(new_shape).sum(axis = -1)
        new_labels = self.values_present.copy()
        new_labels['times'] = np.array(old_times).reshape((len(old_times)//window,window)).mean(axis=-1).tolist()
        return SimData(new_arr, new_labels, self.axis_order)

def plotprep(log = '',
             n_figs: int = 1):
    if n_figs > 1:
        fig, ax = plt.subplots(n_figs,1, figsize = (15,10*n_figs))
    else:
        fig, ax = plt.subplots(1,1)
    axs = force_iterable(ax)
    assert fig is not None
    for ax in axs:
        if 'x' in log:
            ax.set_xscale('log')
        if 'y' in log:
            ax.set_yscale('log')
    return fig, axs

#%%

def hists_at_time(data: SimData,
                  time: float,
                  filters: list[dict],
                  log: str = '',
                  legend_labels: Optional[list[str]] = None,
                  bins = 30,
                  density = False,
                  alpha: float = 0.5,
                  figsavename: Optional[str] = None):

    fig, axs = plotprep(log)
    if legend_labels is None:
        for i, filter in enumerate(filters):
            print(f'Dataset {i+1}: {filter}')
        legend_labels = [f'Dataset {i+1}' for i in range(len(filters))]
    assert len(legend_labels) == len(filters)
    summed_axes = tuple([i for i,axistype in enumerate(data.axis_order) if axistype != 'sims'])
    time = min(data.values_present['times'], key=lambda x:abs(x-time))
    for i,filter in enumerate(filters):
        filter['times'] = [time]
        values = data[filter].array.sum(axis=summed_axes)
        
        total_filter = filter.copy()
        del total_filter['compartments']
        total = data[total_filter].array.sum(axis=summed_axes)    
        values = values/total
        
        axs[0].hist(values, bins = bins, label = legend_labels[i], density = density, alpha=alpha)
    axs[0].set_title(f'Distributions at time {round(time)} days')
    axs[0].set_xlabel('Number of People in Category')
    ylabel = 'Probability Density' if density else 'Frequency'
    axs[0].set_ylabel(ylabel)
    axs[0].legend()
    show_fig(fig, figsavename)

def plot_avg_vals(datasets: Union[SimData, dict[str,SimData]],
                  filters: list[dict],
                  x_axis_type: str = 'times',
                  log: str = '',
                  error_bars: str = 'std',
                  filter_labels: Optional[list[str]] = None,
                  doubling_time: Optional[float] = None,
                  figsavename: Optional[str] =None):
    assert x_axis_type in ['times', 'scaled_times', 'total_infections']
    assert error_bars in ['std', 'IQR', 'None']
    fig, axs = plotprep(log)
    ax = axs[0]
    # if legend_labels is None:
    #     legend_labels = [f'Dataset {i+1}' for i in range(len(filters))]
    #     for i, filter in enumerate(filters):
    #         print(f'Dataset {i+1}: {filter}')
    caption = ''
    if filter_labels is None:
        filter_labels = [str(x) for x in range(1,len(filters)+1)]

    is_dict = isinstance(datasets, dict)
    if is_dict:
        dataset_labels = list(datasets.keys())
        datalist = list(datasets.values())
    else:
        dataset_labels = ['']
        datalist = [datasets]

    assert len(filter_labels) == len(filters)
    assert len(dataset_labels) == len(datalist)
    
    for x, filter in enumerate(filters):
        str_filter = dict_to_str(filter)
        caption += f'Filter {x+1}: {str_filter}\n'
    if is_dict:
        caption += '\n'
        for x, dataset_label in enumerate(dataset_labels):
            caption += f'Dataset {x+1}: {dataset_label}\n'
        caption = caption[:-1]
        


    colors = ['blue', 'red', 'green', 'fuchsia', 'dimgrey', 'yellow','darkviolet', 'darkorange']
    ecolors = ['cornflowerblue', 'lightcoral', 'palegreen', 'lightpink', 'lightgrey', 'lemonchiffon', 'thistle', 'navajowhite']
    linecount, count = 1,1
    total_lines = len(filters) * len(datalist)
    for d, dataset in enumerate(datalist):
        if x_axis_type == 'total_infections':
            total_infections = dataset[create_filter(datatypes=['municipal'],compartments=['E','I','R'])]
            total_infections = total_infections.sum(axis = ('cities','datatypes','groups','compartments'))
            x_axis = total_infections.mean(axis='sims').array
            x_std = total_infections.std(axis='sims').array
        elif x_axis_type == 'times':
            x_axis = np.array(dataset.values_present['times'])
            x_std = 0
        else:
            assert isinstance(doubling_time, float), 'Specify Doubling time to use scaled_times!'
            x_axis = np.array(dataset.values_present['times']) / doubling_time
            x_std = 0

        for f, filter in enumerate(filters):
            values = dataset[filter].sum(axis = ('cities','datatypes','groups','compartments'))

            total_filter = filter.copy()
            del total_filter['compartments']
            total = dataset[total_filter].sum(axis = ('cities','datatypes','groups','compartments'))
            values = values/ (total + 1e-10)

            assert isinstance(values, SimData)
            means = values.mean(axis = 'sims').array
            stds = values.std(axis = 'sims').array
            lower_quartile, median, upper_quartile = values.quartiles(axis='sims',SimData_out=False)

            label = f'Dataset {d+1}, Filter {f+1}'
            if error_bars == 'std':
                ax.errorbar(x_axis,means,yerr=stds, xerr = x_std, label = label, color = colors[linecount], ecolor = ecolors[linecount])
            elif error_bars == 'IQR':
                ax.errorbar(x_axis,median,yerr=np.array([median - lower_quartile, upper_quartile - median]), label = label + ' median', color = colors[linecount], ecolor = ecolors[linecount])
                ax.plot(x_axis,means, '--', color = colors[linecount])
                ax.plot()
            else:
                ax.plot(x_axis,means, color = colors[linecount], label = label)
            print(f'{count}/{total_lines}')
            linecount += 1
            count += 1
            linecount %= len(colors)

    if x_axis_type == 'total_infections':
        xlabel = f'Total Infections\n\n{caption}'
    elif x_axis_type == 'times':
        xlabel = f'Time (days)\n\n{caption}'
    else:
        xlabel = f'Doubling Times\n\n{caption}'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'Fraction of population')
    if error_bars == 'IQR':
        plt.plot([],[],linestyle = '--',color='black',label = 'Means')
    ax.legend()
    show_fig(fig, figsavename)

def plot_infection_ratio(dataset: SimData,
                         filters: list[dict],
                         x_axis_type: str = 'times',
                         max_first: float = 0.1,
                         min_first: Optional[float] = None,
                         n_points: int = 100,
                         log: str = '',
                         error_bars: str = 'std',
                         doubling_time: Optional[float] = None,
                         figsavename: Optional[str] = None):
    #Filters in format: denominator, numerator
    assert x_axis_type in ['times', 'scaled_times', 'total_infections']
    assert error_bars in ['std', 'IQR', 'None']
    fig, axs = plotprep(log)
    ax = axs[0]
    if x_axis_type == 'total_infections':
        total_infections = dataset[create_filter(datatypes=['municipal'],compartments=['E','I','R'])]
        total_infections = total_infections.sum(axis = ('cities','datatypes','groups','compartments'))
        x_axis = total_infections.mean(axis='sims').array
        x_std = total_infections.std(axis='sims').array
    elif x_axis_type == 'times':
        x_axis = np.array(dataset.values_present['times'])
        x_std = 0
    else:
        assert isinstance(doubling_time, float), 'Specify Doubling time to use scaled_times!'
        x_axis = np.array(dataset.values_present['times']) / doubling_time
        x_std = 0
    data_1, data_2 = (dataset[f] for f in filters)
    values_array = data_2.array/(data_1.array + 1e-10)
    values = SimData(values_array, data_2.values_present)
    assert isinstance(values, SimData)
    means = values.mean(axis = 'sims').array.squeeze()
    stds = values.std(axis = 'sims').array.squeeze()
    lower_quartile, median, upper_quartile = values.quartiles(axis='sims',SimData_out=False)
    if error_bars == 'std':
        ax.errorbar(x_axis,means,yerr=stds, xerr = x_std)
    elif error_bars == 'IQR':
        ax.errorbar(x_axis,median,yerr=np.array([median - lower_quartile, upper_quartile - median]), label = 'median')
        ax.plot(x_axis,means, '--', color = 'black', label = 'mean')
        ax.plot()
    else:
        ax.plot(x_axis,means)
    if x_axis_type == 'total_infections':
        xlabel = 'Total Infections'
    elif x_axis_type == 'times':
        xlabel = 'Time (days)'
    else:
        xlabel = 'Doubling Times'
    ax.set_xlabel(xlabel)
    same_categories = {k:v for k,v in filters[0].items() if k in filters[1].keys() and v == filters[1][k]}
    uniques = [{k:v for k,v in f.items() if k not in same_categories.keys()} for f in filters]
    ylabel = f'{dict_to_str(same_categories)}. Ratio of {dict_to_str(uniques[1])} / {dict_to_str(uniques[0])}'
    ax.set_ylabel(ylabel)
    ax.legend()
    show_fig(fig, figsavename)

def joint_times_distribution(data: SimData,
                       filters: list[dict],
                       threshold: float,
                       log: str = '',
                       legend_labels: Optional[list[str]] = None,
                       bins = 30,
                       density = False,
                       alpha: float = 0.5,
                       figsavename: Optional[str] = None):
    assert len(filters) == 2
    if legend_labels is None:
        for i, filter in enumerate(filters):
            print(f'Dataset {i+1}: {filter}')
        legend_labels = [f'Dataset {i+1}' for i in range(len(filters))]
    assert len(legend_labels) == len(filters)
    fig, axs = plotprep(log)
    ax = axs[0]
    
    filtered = [data[filter].sum(axis=('cities','datatypes','groups','compartments')).array for filter in filters]

    total_filters = [filter.copy() for filter in filters]
    for f in total_filters:
        del f['compartments']
    totals = [data[filter].sum(axis=('cities','datatypes','groups','compartments')).array for filter in total_filters]
    fractions = [a/b for a,b in zip(filtered,totals)]
    steps = [find_first_index(arr>threshold) for arr in fractions]
    x,y = tuple([np.array(data.values_present['times'])[step] for step in steps ])
    ax.hist2d(x, y, bins=(bins,bins), cmap = plt.cm.jet)
    ax.set_xlabel(f'Times in {legend_labels[0]}')
    ax.set_ylabel(f'Times in {legend_labels[1]}')
    ax.set_title(f'Time(days) till fraction in datasets reach {threshold}')
    show_fig(fig, figsavename)

def times_until_threshold(data: SimData,
                          filter: dict,
                          threshold: float):
    filtered = data[filter].sum(axis=('cities','datatypes','groups','compartments')).array
    total_filter = filter.copy()
    del total_filter['compartments']
    totals = data[total_filter].sum(axis=('cities','datatypes','groups','compartments')).array
    fraction = filtered/np.maximum(totals, 1e-10)
    return np.array(data.values_present['times'])[find_first_index(fraction>threshold)]

def time_diff_at_threshold(data: SimData,
                           filters: list[dict],
                           threshold1: float,
                           threshold2: Optional[float] = None):
    assert len(filters) == 2
    if threshold2 is None:
        threshold2 = threshold1
    thresholds = (threshold1,threshold2)
    filtered = [data[filter].sum(axis=('cities','datatypes','groups','compartments')).array for filter in filters]

    total_filters = [filter.copy() for filter in filters]
    for f in total_filters:
        del f['compartments']
    totals = [data[filter].sum(axis=('cities','datatypes','groups','compartments')).array for filter in total_filters]
    fractions = [a/np.maximum(b, 1e-10) for a,b in zip(filtered,totals)]
    steps = [find_first_index(arr>threshold) for arr, threshold in zip(fractions,thresholds)]
    x,y = tuple([np.array(data.values_present['times'])[step] for step in steps ])
    return y - x

def different_thresholds_diffs_data(data: SimData,
                                    filters: list[dict],
                                    thresholds1: np.ndarray = 10**np.linspace(-6,-1,24),
                                    thresholds2: Optional[np.ndarray] = None):
    if thresholds2 is None:
        thresholds2 = thresholds1
    times = np.zeros((len(thresholds1), len(thresholds2), 2, len(data.values_present['sims'])))
    for i,threshold in tqdm(enumerate(thresholds1)):
        times[i,:,0] = times_until_threshold(data, filters[0], threshold)
    for i,threshold in tqdm(enumerate(thresholds2)):
        times[:,i,1] = times_until_threshold(data, filters[1], threshold)
    return times[:,:,1], times[:,:,0]

def different_thresholds_diffs(data: SimData,
                               filters: list[dict],
                               x_thresholds: np.ndarray = 10**np.linspace(-6,-1,24),
                               y_thresholds: Optional[np.ndarray] = None, 
                               log = 'xy',
                               include_line = True,
                               zero_centre = True,
                               figsavename: Optional[str] = None):
    fig, axs = plotprep(log)
    if y_thresholds is None:
        y_thresholds = x_thresholds
    ax = axs[0]
    a,b = different_thresholds_diffs_data(data, filters, x_thresholds, y_thresholds)
    diffs = a - b
    mean_diffs = diffs.mean(axis = -1)
    positions = {'x': x_thresholds, 'y': y_thresholds}
    edges = {}
    for axis in ['x','y']:
        if axis in log: 
            ps = np.log10(positions[axis])
        else:
            ps = positions[axis]
        edge = [(ps[i]+ps[i+1])/2 for i in range(len(ps)-1)]
        edge = np.array([2*ps[0]-edge[0]] + edge + [2*ps[-1]-edge[-1]])
        if axis in log:
            edge = 10 ** edge
        edges[axis] = edge
    y,x = np.meshgrid(edges['y'], edges['x'])
    # a = np.ones((len(x)-1, len(y)-1))
    # im = ax.pcolor(x, y, a, cmap = 'bwr')
    # print(mean_diffs)
    max_range = max(np.abs(np.min(mean_diffs)), np.abs(np.max(mean_diffs)))
    if zero_centre:
        im = ax.pcolor(x, y, mean_diffs, cmap = 'seismic', vmin = -max_range, vmax = max_range)
    else:
        im = ax.pcolor(x, y, mean_diffs, cmap = 'hot')
    minval = max(x_thresholds.min(), y_thresholds.min())
    maxval = min(x_thresholds.max(), y_thresholds.max())
    xs = np.logspace(np.log10(minval), np.log10(maxval), 3)
    if include_line:
        ax.plot(xs,xs, color = 'black')
    ax.set_xlabel(f'Threshold Fraction for {dict_to_str(filters[0])}')
    ax.set_ylabel(f'Threshold Fraction for {dict_to_str(filters[1])}')
    ax.set_title(f'Time to detection in ({dict_to_str(filters[0])}) - Time to detection ({dict_to_str(filters[1])})')
    fig.colorbar(im, ax=ax)
    
    show_fig(fig, figsavename)

def threshold_ratio_diffs(data: SimData,
                          filters: list[dict],
                          filter_names: list[str] = ['Municipal', 'Airplane'],
                          n_points: int = 30,
                          xlims_log: list[Union[tuple,float]] = [(-1.5,-0.7)],
                          ylims_log: Union[tuple,int] = (-0.7,-6),
                          log = 'x',
                          error_bars = 'std',
                          figsavename: Optional[str] = None):
    fig, axs = plotprep(log)
    assert error_bars in ['std', 'IQR', 'None']
    ax = axs[0]
    for xlimlog in xlims_log:
        x_points = 10**np.linspace(xlimlog[0],xlimlog[1],n_points) if isinstance(xlimlog,tuple) else np.ones(n_points) * 10 ** xlimlog
        y_points = 10**np.linspace(ylims_log[0],ylims_log[1],n_points) if isinstance(ylims_log,tuple) else np.ones(n_points) * 10 ** ylims_log
        ratios = y_points / x_points
        diffs = np.zeros((n_points, len(data.values_present['sims'])))
        for i in tqdm(range(n_points)):
            diffs[i] = time_diff_at_threshold(data,filters,x_points[i], y_points[i])
        means = diffs.mean(axis=1)
        lower_quartiles, medians, upper_quartiles = quartiles(diffs, axis = 1)
        stds = diffs.std(axis=1)
        if error_bars == 'None':
            if isinstance(xlimlog, float):
                threshold = 10**xlimlog
                label = f'{filter_names[1]} Threshold = {float(f"{threshold:.1g}"):g}'
                ax.plot(ratios, means,label=label)
            else:
                ax.plot(ratios, means)
        elif error_bars == 'std':
            if isinstance(xlimlog, float):
                threshold = 10**xlimlog
                label = f'{filter_names[1]} Threshold = {float(f"{threshold:.1g}"):g}'
                ax.errorbar(ratios, means,stds,label=label)
            else:
                ax.errorbar(ratios, means,stds)
        else:
            if isinstance(xlimlog, float):
                threshold = 10**xlimlog
                label = f'Median, {filter_names[1]} Threshold = {float(f"{threshold:.1g}"):g}'
            else:
                label = 'Median'
            ax.errorbar(ratios, medians, [medians-lower_quartiles, upper_quartiles-medians], label = label)
            ax.plot(ratios, means, '--', color='black')
    ax.set_xlabel(f'Ratio of {filter_names[1]} threshold to {filter_names[1]} threshold')
    ax.set_ylabel(f'Advantage of using {filter_names[1]} detection (days)')
    if error_bars == 'IQR':
        ax.plot('', '--', color = 'black', label = 'Means')
    if len(xlims_log) > 1 or error_bars == 'IQR':
        ax.legend()
    show_fig(fig, figsavename)

def time_diff_at_threshold_hist(data: SimData,
                                filters: list[dict],
                                threshold: float,
                                log: str = '',
                                legend_labels: Optional[list[str]] = None,
                                bins = 30,
                                density = False,
                                daily = False,
                                figsavename: Optional[str] = None):
    assert len(filters) == 2
    if legend_labels is None:
        # for i, filter in enumerate(filters):
        #     print(f'Dataset {i+1}: {filter}')
        legend_labels = [f'Dataset {i+1}' for i in range(len(filters))]
    assert len(legend_labels) == len(filters)
    fig, axs = plotprep(log)
    ax = axs[0]
    
    to_plot = time_diff_at_threshold(data, filters, threshold)
    if not daily:
        # to_plot = y-x
        ax.hist(to_plot, bins=bins, density=density)
        ylabel = 'Probability Density' if density else 'Frequency'
        ax.set_ylabel(ylabel)
    else:
        to_plot = np.round(to_plot).astype(np.int64)
        options = list(set(to_plot))
        amounts = np.bincount(to_plot,minlength = len(options))
        amounts = amounts[-len(options):]
        pos = np.arange(len(options))
        options = [str(option) for option in options]
        ax.bar(pos,amounts)
        ax.set_xticks(pos, options)
        ax.set_ylabel('Frequency')
    filters_str = [dict_to_str(filter) for filter in filters]
    ax.set_xlabel(f'Time for {legend_labels[1]} - time for {legend_labels[0]} (days) \n\nDataset 1: {filters_str[0]}\nDataset 2: {filters_str[1]}')
    ax.set_title(f'Difference between time to reach {threshold} in the datasets')
    show_fig(fig, figsavename)

def differences_vs_threshold_data(datasets: dict[str,SimData],
                     filters: list[dict],
                     thresholds: Union[list[float], np.ndarray] = 10**np.linspace(-4,-0.7,20)):
    assert len(filters) == 2
    out = np.zeros((5,len(datasets.values()), len(thresholds)))
    for i, dataset in enumerate(datasets.values()):
        for j,threshold in tqdm(enumerate(thresholds)):
            differences = time_diff_at_threshold(dataset, filters, threshold)
            out[0,i,j] = differences.mean()
            out[4,i,j], out[1,i,j], out[3,i,j] = quartiles(differences)#type: ignore
            out[2,i,j] = differences.std() #type: ignore
    return out

def differences_vs_threshold(datasets: dict[str,SimData],
                             calculated_data: np.ndarray,
                             filters: list[dict],
                             thresholds: Union[list[float], np.ndarray] = 10**np.linspace(-4,-0.7,21),
                             log: str = 'x',
                             error_bars: str = 'std',
                             figsavename: Optional[str] = None):
    assert error_bars in ['std', 'IQR', 'None']
    assert len(filters) == 2
    assert len(calculated_data) == 5
    fig, axs = plotprep(log)
    colors = ['blue', 'red', 'green', 'fuchsia', 'dimgrey', 'yellow','darkviolet', 'darkorange']
    ecolors = ['cornflowerblue', 'lightcoral', 'palegreen', 'lightpink', 'lightgrey', 'lemonchiffon', 'thistle', 'navajowhite']
    ax = axs[0]
    legend_labels = list(datasets.keys())
    means, medians, stds, upper_quartiles, lower_quartiles = tuple(calculated_data)

    for i in tqdm(range(len(datasets))):
        col_idx = i%len(colors)
        if error_bars == 'None':
            ax.plot(thresholds, means[i], label = legend_labels[i])
        else:
            errors = np.array([medians[i]-lower_quartiles[i], upper_quartiles[i]-medians[i]]) if error_bars == 'IQR' else stds[i] #type: ignore
            y = means[i] if error_bars == 'std' else medians[i] #type: ignore
            plt.errorbar(thresholds, y, errors, color = colors[col_idx], ecolor = ecolors[col_idx], label = legend_labels[i])
    filters_str = [dict_to_str(filter) for filter in filters]
    ax.set_xlabel(f'Threshold\n\nTime Difference between:\nDataset 1: {filters_str[0]}\nDataset 2: {filters_str[1]}')
    ax.set_ylabel('Time Difference (days)')
    ax.legend()
    show_fig(fig, figsavename)

def differences_vs_variable(datasets: dict[str,SimData],
                             variables: list[int],
                             calculated_data: np.ndarray,
                             filters: list[dict],
                             old_thresholds: np.ndarray,
                             new_thresholds: np.ndarray,
                             log: str = 'x',
                             error_bars: str = 'std',
                             legend_labels: Optional[list[str]] = None,
                             x_name: str = 'Daily Mixing Rate',
                             figsavename: Optional[str] = None):
    assert error_bars in ['std', 'IQR', 'None']
    assert len(filters) == 2
    assert len(variables) == len(datasets)
    assert len(calculated_data) == 5
    fig, axs = plotprep(log)
    colors = ['blue', 'red', 'green', 'fuchsia', 'dimgrey', 'yellow','darkviolet', 'darkorange']
    ecolors = ['cornflowerblue', 'lightcoral', 'palegreen', 'lightpink', 'lightgrey', 'lemonchiffon', 'thistle', 'navajowhite']
    ax = axs[0]
    threshold_idxs = [np.argmin(np.abs(old_thresholds - t)) for t in new_thresholds]
    thresholds = old_thresholds[threshold_idxs]
    print(thresholds)
    calculated_data = calculated_data[...,np.array(threshold_idxs)]
    means, medians, stds, upper_quartiles, lower_quartiles = tuple(calculated_data)

    if legend_labels is None:
        legend_labels = ["Threshold: {:.0e}".format(threshold) for threshold in thresholds]
    for i, threshold in enumerate(thresholds):
        col_idx = i%len(colors)
        if error_bars == 'None': 
            ax.plot(variables, means[:,i], label = legend_labels[i])
        else:
            errors = np.array([medians[:,i]-lower_quartiles[:,i], upper_quartiles[:,i]-medians[:,i]]) if error_bars == 'IQR' else stds[:,i] #type: ignore
            y = means[:,i] if error_bars == 'std' else medians[:,i] #type: ignore
            plt.errorbar(variables, y, errors, color = colors[col_idx], ecolor = ecolors[col_idx], label = legend_labels[i])
    filters_str = [dict_to_str(filter) for filter in filters]
    ax.set_xlabel(f'{x_name}\n\nTime Difference between:\nDataset 1: {filters_str[0]}\nDataset 2: {filters_str[1]}')
    ax.set_ylabel('Time Difference (days)')
    ax.legend()
    show_fig(fig, figsavename)

def differences_vs_variable_final_img(datasets: dict[str,SimData],
                             variables: list[int],
                             calculated_data: np.ndarray,
                             filters: list[dict],
                             old_thresholds: np.ndarray,
                             new_thresholds: np.ndarray,
                             log: str = 'x',
                             error_bars: str = 'std',
                             legend_labels: Optional[list[str]] = None,
                             x_name: str = 'Mixing Rate per Day',
                             include_legend = False,
                             with_plateau = True,
                             figsavename: Optional[str] = None):
    assert error_bars in ['std', 'IQR', 'None']
    assert len(filters) == 2
    assert len(variables) == len(datasets)
    assert len(calculated_data) == 5
    fig, axs = plotprep(log)
    colors = ['blue', 'red', 'green', 'fuchsia', 'dimgrey', 'yellow','darkviolet', 'darkorange']
    ecolors = ['cornflowerblue', 'lightcoral', 'palegreen', 'lightpink', 'lightgrey', 'lemonchiffon', 'thistle', 'navajowhite']
    ax = axs[0]
    threshold_idxs = [np.argmin(np.abs(old_thresholds - t)) for t in new_thresholds]
    thresholds = old_thresholds[threshold_idxs]
    calculated_data = calculated_data[...,np.array(threshold_idxs)]
    means, medians, stds, upper_quartiles, lower_quartiles = tuple(calculated_data)

    if legend_labels is None:
        legend_labels = ["Threshold: {:.0e}".format(threshold) for threshold in thresholds]
    for i, threshold in enumerate(thresholds):
        x = variables if with_plateau else [v for v in variables if v > 1/threshold]

        col_idx = i%len(colors)
        if error_bars == 'None': 
            ax.plot(x, means[:,i][-len(x):], label = legend_labels[i])
        else:
            errors = np.array([medians[:,i]-lower_quartiles[:,i], upper_quartiles[:,i]-medians[:,i]]) if error_bars == 'IQR' else stds[:,i] #type: ignore
            y = means[:,i] if error_bars == 'std' else medians[:,i] #type: ignore
            plt.errorbar(x, y[-len(x):], errors[-len(x):], color = colors[col_idx], ecolor = ecolors[col_idx], label = legend_labels[i])
    filters_str = [dict_to_str(filter) for filter in filters]
    ax.set_xlabel(x_name)
    ax.set_ylabel('Advantage of using Airplane Detection (days)')
    if include_legend:
        ax.legend()
    show_fig(fig, figsavename)

def create_filter(cities: Optional[Iterable[Union[str,int]]]=None,
                  datatypes: Optional[Iterable[Union[str,int]]]=None,
                  groups: Optional[Iterable[Union[str,int]]]=None,
                  compartments: Optional[Iterable[Union[str,int]]]=None,
                  sims: Optional[Iterable[Union[str,int]]]=None,
                  times: Optional[Iterable[Union[str,int]]]=None):
    return {key:val for key, val in locals().items() if val is not None}

def create_arrival_municipal_filter(city = 1, groups = None, compartments = 'I'):
    datatypes = ['arrivals', 'municipal']
    return [create_filter([city], [datatype], groups, list(compartments)) for datatype in datatypes]

#%%
class Travel(City):
    def __init__(self,
                 cities: list[City],
                 mixmatrix: np.ndarray,
                 city_names: Optional[list[str]] = None):
        self.cities = cities
        self.n_cities = len(cities)
        self.city_names = list(range(self.n_cities)) if city_names is None else city_names
        for i,city in enumerate(self.cities):
            city.name = self.city_names[i]
        self.mixmatrix = mixmatrix.astype(np.int64)
        assert np.all(mixmatrix == mixmatrix.T), "mixmatrix must be symmetrical!"
        assert np.trace(mixmatrix) == 0, "mixmatrix must be traceless!"
        self.n_groups = self.cities[0].n_groups
        self.compartments = list(self.cities[0].compartments)
        self.I0s = []

        self.colors = ['blue', 'red', 'green', 'fuchsia', 'dimgrey', 'yellow','darkviolet', 'darkorange']
        self.ecolors = ['cornflowerblue', 'lightcoral', 'palegreen', 'lightpink', 'lightgrey', 'lemonchiffon', 'thistle', 'navajowhite']
        self.ncolors = len(self.colors)
  
    def reset_parameters(self):
        for i,city in enumerate(self.cities):
            city.reset_parameters(I0 = self.I0s[i],
                                  n_sims = self.n_sims,
                                  simulation_steps = self.simulation_steps)
    
    def step(self,
            beta:float,
            delta_t: float,
            p_infectious: float,
            p_recovery: float,
            simulation_step: int):
        for i,city in enumerate(self.cities):
            city.step_internal(beta,
                               delta_t,
                               p_infectious,
                               p_recovery,
                               simulation_step)
        for i, city_i in enumerate(self.cities[:-1]):
            for j, city_j in enumerate(self.cities[i+1:]):
                j += i+1
                mix = self.mixmatrix[i,j]
                if mix == 0:
                    continue
                travel_ij = city_i.select_travellers(mix, simulation_step)
                travel_ji = city_j.select_travellers(mix, simulation_step)
                
                city_i.municipal[...,simulation_step] += travel_ji - travel_ij
                city_i.arrivals[...,simulation_step] += travel_ji
                city_i.departures[...,simulation_step] += travel_ij
                
                city_j.municipal[...,simulation_step] += travel_ij - travel_ji
                city_j.arrivals[...,simulation_step] += travel_ij
                city_j.departures[...,simulation_step] += travel_ji
    
    def multiple_sims(self,
                      delta_t: float,
                      epidemic_time: Union[int,float],
                      disease: Disease,
                      I0s: Union[int,float,list[int],list[float],np.ndarray],
                      n_sims: int = 100,
                      moving_avg: bool = False):
        if isinstance(I0s, np.ndarray):
            assert I0s.dtype == np.int64
        elif isinstance(I0s, int):
            I0s = np.array([I0s] + [0] * (len(self.cities) - 1))
        elif isinstance(I0s, float):
            I0s = np.array([I0s * self.cities[0].N0] + [0] * (len(self.cities) - 1))
        elif isinstance(I0s, list):
            I0s = [I * self.cities[j].N0 if isinstance(I,float) else I for j,I in enumerate(I0s)]
            I0s = np.array(I0s)
        self.I0s = I0s
        assert (1 / delta_t) % 1 == 0, "1/delta_t must be an integer"
        self.start_city = self.cities[[I0 != 0 for I0 in I0s].index(True)]
        p_recovery = 1 - np.exp( - delta_t * disease.gamma)
        p_infectious = 1 - np.exp( - delta_t * disease.delta)
        self.simulation_steps = int(epidemic_time / delta_t)
        self.n_sims = n_sims
        self.times = np.linspace(0, epidemic_time, self.simulation_steps)
        self.scaled_times = self.times /disease.doubling_time
        self.delta_t = delta_t
        for city in self.cities:
            city.delta_t = delta_t
            city.n_sims = n_sims
            city.epidemic_time = epidemic_time
            city.times = self.times
            city.scaled_times = self.scaled_times
            city.simulation_steps = self.simulation_steps
            city.disease = disease
        self.reset_parameters()
        for simulation_step in tqdm(range(1,self.simulation_steps)):
            self.step(disease.beta,
                      delta_t,
                      p_infectious,
                      p_recovery,
                      simulation_step)
        for city in tqdm(self.cities):
            city.daily_flight_data(moving_avg = moving_avg)
        
        #make and return SimData
        n_datatypes = 5 if moving_avg else 3
        out_arr = np.zeros((self.n_cities, n_datatypes, self.n_groups, len(self.compartments), self.n_sims, self.simulation_steps))
        if moving_avg:
            for i,city in enumerate(self.cities):
                out_arr[i] = np.array([city.municipal,
                                    city.arrivals,
                                    city.departures,
                                    city.arrivals_moving_avg,
                                    city.departures_moving_avg])
            all_labels = {'cities': self.city_names,
                        'datatypes': ['municipal', 'arrivals', 'departures', 'arrivals_moving_avg', 'departures_moving_avg'],
                        'groups': self.cities[0].groups,
                        'compartments': self.compartments,
                        'sims': list(range(self.n_sims)),
                        'times': self.times}
        else:
            for i,city in enumerate(self.cities):
                out_arr[i] = np.array([city.municipal,
                                    city.arrivals,
                                    city.departures])
            all_labels = {'cities': self.city_names,
                        'datatypes': ['municipal', 'arrivals', 'departures'],
                        'groups': self.cities[0].groups,
                        'compartments': self.compartments,
                        'sims': list(range(self.n_sims)),
                        'times': self.times}

        return SimData(out_arr, all_labels)

    def __str__(self):
        out = ''
        for i, city in enumerate(self.cities):
            out += f'City: {self.city_names[i]}\n'
            out += str(city) + '\n'
        if self.I0s is not None:
            out += f'I0s: {self.I0s}\n'
        out += f'Mixmatrix: {self.mixmatrix}'
        return out
    
    def plot_sims(self,
                  to_shift: bool = True, #change to shift_by: Optional[Union[int,str]] = None, assert shift_by in city.groups
                  included_cities: Optional[list[int]] = None,
                  separate_groups: bool = True,
                  figsavename: Optional[str] = None,
                  moving_avg = False,
                  log=''):

        if included_cities is None:
            included_cities = list(range(len(self.cities)))
            included_names = included_cities
        elif isinstance(included_cities[0],int):
            included_names = included_cities
        elif isinstance(included_cities[0],str):
            included_names = included_cities
            included_cities = [i for i, name in enumerate(self.city_names) if name in included_cities]
        assert isinstance(included_names,list) #type: ignore
        shift_index = None
        if to_shift:
            shift_index = self.start_city.municipal[:,self.start_city.I_index].sum(axis=0).argmax(axis = 1)
        # for i, city in enumerate(self.cities):
        for j,i in enumerate(included_cities):
            city = self.cities[i]
        # if i in included_cities:
            if figsavename is None:
                savename = None
            else:
                savename = f'{figsavename}_{i}'
            print(f"City {included_names[j]}:")
            city.plot_sims(times = self.times,
                            cityname=i,
                            shift_index = shift_index,
                            separate_groups=separate_groups,
                            moving_avg=moving_avg,
                            log=log,
                            figsavename = savename)
            print('\n')
    
    def __call__(self,
                 delta_t: float,
                 epidemic_time: Union[int,float],
                 disease: Disease,
                 I0s: Union[int,float,list[int],list[float], np.ndarray],
                 n_sims: int = 100,
                 moving_avg = True):
        return self.multiple_sims(delta_t, epidemic_time, disease, I0s, n_sims, moving_avg = moving_avg)
#%%
class ReturnHomeCity(City):
    def __init__(self,
                 city_names: list[Union[int,str]],
                 this_city_name: Union[int,str],
                 mixmatrix: np.ndarray,
                 N0: int = 10**6,
                 group_LR: float = 0.1,
                 trip_length: int = 10,
                 p_go_home: float = 0.9,
                 p_continue_travel: float = 0.05,
                 compartments: str = 'SEIR'):
        self.city_index = city_names.index(this_city_name)
        self.n_cities = len(city_names)
        self.NOs = np.ones(len(city_names),np.int64) * round(mixmatrix[self.city_index] * trip_length)
        self.N0s[self.city_index] = N0 - self.NOs[:-1].sum()
        assert self.N0s.sum() == N0
        self.groups = city_names
        self.n_groups = len(city_names)
        group_LR_matrix = homogeneous_LR_matrix(self.n_groups, group_LR)
        super().__init__(self.N0s, self.groups, compartments, group_LR_matrix)
        self.trip_length = trip_length
        self.p_go_home = p_go_home
        self.p_continue_travel = p_continue_travel
        
    def initial_conditions(self):
        #puts all initial infected people into the compartment of people who are in their home city
        self.municipal[self.city_index,2,:,0] = self.I0
    
    def select_travellers(self,
                          mix: int,
                          destination_index: int,
                          simulation_step: int):
        p_travel = 1 - np.exp(- self.delta_t * mix / self.N0s[self.city_index])
        p_return_home = 1 - np.exp(- self.delta_t * self.p_go_home / (self.trip_length * self.N0s[self.city_index]))
        p_onward_travel = 1 - np.exp(- self.delta_t * self.p_continue_travel / (self.trip_length * self.N0s[self.city_index]))

        to_travel = np.random.binomial(self.municipal[...,simulation_step],p_onward_travel)
        to_travel[self.city_index] = np.random.binomial((self.municipal[self.city_index,...,simulation_step]), p_travel)
        to_travel[destination_index] = np.random.binomial((self.municipal[destination_index,...,simulation_step]), p_return_home)
        return to_travel 

    def select_settlers(self, simulation_step: int):
        p_settle = 1 - np.exp(- self.delta_t * (1-self.p_go_home-self.p_continue_travel) / (self.trip_length * self.N0s[self.city_index] * (self.n_cities-1)))
        to_settle = np.random.binomial(self.municipal[...,simulation_step], p_settle)
        self.municipal[...,simulation_step] -= to_settle
        self.municipal[self.city_index,...,simulation_step] += to_settle.sum(axis=0)

class ReturnHomeTravel(Travel):
    def __init__(self):
        pass

    def step(self,
            beta:float,
            delta_t: float,
            p_infectious: float,
            p_recovery: float,
            simulation_step: int):
        for i,city in enumerate(self.cities):
            city.step_internal(beta,
                               delta_t,
                               p_infectious,
                               p_recovery,
                               simulation_step)
        for i, city_i in enumerate(self.cities[:-1]):
            assert isinstance(city_i, ReturnHomeCity)
            for j, city_j in enumerate(self.cities[i+1:]):
                j += i+1
                mix = self.mixmatrix[i,j]
                if mix == 0:
                    continue
                travel_ij = city_i.select_travellers(mix, j, simulation_step)
                travel_ji = city_j.select_travellers(mix, i, simulation_step)
                
                city_i.municipal[...,simulation_step] += travel_ji - travel_ij
                city_i.arrivals[...,simulation_step] += travel_ji
                city_i.departures[...,simulation_step] += travel_ij
                
                city_j.municipal[...,simulation_step] += travel_ij - travel_ji
                city_j.arrivals[...,simulation_step] += travel_ij
                city_j.departures[...,simulation_step] += travel_ji
            city_i.select_settlers(simulation_step)

#%%
class HomogeneousNetwork(Travel):
    def __init__(self,
                 citytype: Type,
                 n_cities: int,
                 mixnumber: int,
                 *args, **kwargs):
        cities = []
        for i in range(n_cities):
            cities.append(citytype(*args, **kwargs))
        mixmatrix = (np.ones((n_cities,n_cities)) - np.identity(n_cities)).astype(np.int64) * mixnumber
        super().__init__(cities, mixmatrix)

