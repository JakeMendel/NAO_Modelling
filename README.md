# High Level Structure

`functions_and_classes.py` is the file that contains all the classes and functions. 

`images_code.py` has example code to generate 3 of the figures that I use in my report.

`generic_model.ipynb` is a jupyter notebook with various scripts for generating other useful plots.

# Dependencies
I have used
- matplotlib version 3.7.1
- numpy version 1.24.1
- tqdm version 4.62.3
- fancy_einsum version 0.0.3

# The Model

I use a discrete time stochastic network SEIR model to simulate an epidemic. The epidemic starts in a single city, with some number of people initially chosen to be infectious (or exposed). Each timestep:
- Infectious people make susceptible people exposed
- Exposed people become infectious
- Infectious people are removed (they recover with immunity or they die)
- Some people travel from each city to the other cities

Eventually, some of the people who are travelling to other cities from the city with the first outbreak will be exposed or infectious, so they start new outbreaks in the other cities.

The details of who travels and when is variable:
- In a `BasicCity` network, everyone is equally likely to travel all the time.
- In a `FrequentFlierCity` network, there are two classes of people: normal people and frequent fliers. Frequent fliers (air stewards, businesspeople etc) fly more often than normal, and they are more likely to interact with each other than with other people
- In a `ReturnHomeCity` network, everyone has a home city. When people leave their home city, they have a small chance of staying there and a small chance of travelling on to another city, but a high chance returning to their home city within a week or two.

Each `City` is initialised with the following parameters:
- `N`: the city’s population
- A list groups with length `n_groups`. This is the different types of person in the `City`. In `BasicCity`, groups = [normal]. In `FrequentFlyerCity`, `groups = [normal, frequent_flyers]`. In `ReturnHomeCity`, there is one group for each home city.
- `mixing_LR`: a matrix of size `(n_groups, n_groups)`, with diagonal entries equal to 1 and off-diagonal entries between 0 and 1. The i,jth component is the likelihood of a member of group i having close contact with a member of group j relative to another member of group i.

More sophisticated cities also have extra parameters. In `FrequentFlyerCity`, the other parameters are:
- `frequent_flier_frac`: the fraction of the population that is in the frequent flyer class of people 
- `flying_LR`: The ratio between the likelihood of a frequent flier flying and a normal person flying on a given day.
  - Alternatively, `p_ff` can be specified: the probability that a given person on a plane is a frequent flier. Any two of `frequent_flier_frac, flying_LR, p_ff` can be used to calculate the third according to:
    
    `p_ff = flying_LR * frequent_flier_frac / (flying_LR * frequent_flier_frac + (1 - frequent_flier_frac))`


In `ReturnHomeCity` (not fully implemented yet), the other parameters are:
- `Trip_length`: the average time people spend as travellers in a city that isn’t their home
- `P_go_home`: the probability that the trip ends with returning home
- `p_continue_travel`: the probability that the trip ends with traveling to a 3rd city
- `1-p_go_home-p_continue_travel` is therefore the probability that the travelling ends with the traveller settling permanently in the city they visited.

The class `Travel` is initialised with:
- `cities`: an ordered list of each `City` within the network. In all cases I have studied, all cities are of the same type.
- `mixmatrix`: a symmetric matrix whose i,jth component is the number of daily travellers between city i and city j

After creating an instance of `Travel`, a simulation can be run by either calling `Travel.multiple_sims` or by calling the instance of the class directly. The arguments that can be passed to the simulation are:
- `delta_t`: the size of the timestep.
The simulation is with discrete time, so ideally this time interval should be smaller than $1/2r_{max}$ where $r_{max}$ is the maximum rate of change in the model. In my simulations, I have mostly worked with `delta_t` = 0.04 days, which is valid when there are no rates greater than 12.5/day.
- `epidemic_time`: how long the simulation is run for.
- `disease`: an instance of the `Disease` class with attributes `beta, gamma, delta` which are respectively:
  - the rate at which an infectious person causes susceptible people to become exposed in an immune naive population
  - recovery rate
  - rate at which exposed people become infectious
- `I0s`: the initial number of infected people in each city (0 for all cities except the start city)
- `n_sims`: the number of simulations to run. In each sim, the initial conditions are the same.

At each timestep in the simulation:

1. The epidemic spreads for one timestep in each city:
    
    1. The rate at which susceptible people are infected is calculated for each group in according to: $$r_i = \beta \frac{m_{ij}I_j}{m_{ij}N_j}$$
    Here:
        - $I_i$ is the number of infected people in this city in group $i$ at this timestep (the time dependence is suppressed here)
        - $N_i$ is the number of infected people in this city in group $i$ at this timestep
        - $m_{ij}$ is the matrix `mixing_LR`
        - $\beta$ = `disease.beta`.
    This expression is a weighted average of infection rates from different groups.

    2. Exposed people become infectious at rate `disease.delta`
    3. Infectious people recover at rate `disease.gamma`
    4. All transitions between compartments are assumed to be independent poisson processes. This means that time spent in any compartment is exponentially distributed with rate parameter equal to the transition rate. Hence the probability that any member of a compartment with transition rate r undergoes a transition in this timestep is $1-e^{-rt}$. Using the independent transitions assumption we can simulate the number of transitions from compartment A to compartment B in this timestep by sampling from a binomial distribution: $$n_{A\rightarrow B} \sim \text{Binom}(n_A,1-e^{-r_{A\rightarrow B }t})$$
    Where $n_A$ is the number of people in compartment A.

2. Next, we simulate travel between cities. For all cities $i$ and $j$:
    1. The overall rate of travel from city $i$ to city $j$ is `mixmatrix[i,j]`.
    2. In non-basic cities, the rates of travel will vary by group, so that the total rate of transition from all groups in city $i$ to all groups in city $j$ is `mixmatrix[i,j]`. For example, in `FrequentFlyerCity`, the fraction of travellers that are in the frequent flyer group is the parameter `p_ff`. Hence the travel rate for this group is `mixmatrix[i,j] * p_ff`.
    3. I have not enforced that the number of people in each city is conserved. Therefore the number of people in each city will undergo a random walk over time. To avoid this, the expected number of people to leave city $i$ and group $g$ is scaled by $\frac{N_{ig}(t)}{N_{ig}(0)}$ (where ${N_{ig}(t)}$ is the number of people in group $g$ and city $i$ at time $t$) to provide a pressure to conserve populations in each group and city.
    4. For each group, the number of people who travel at each timestep is calculated in exactly the same way as in step 1(d). The assumptions this implies are:
        - Travelling is a Poisson process, and individuals travel independently (whereas in reality, people travel in batches on planes).
        - Travel is instantaneous
        - Travel rates are independent of compartment (S/E/I/R)
    5. This rate is converted into an array of travellers at different infection stages for each group who travel from city $i$ to $j$.
    6. The number of arrivals and departures in each of the 8 compartments are stored, and the number of people of each type in each city is updated.
3. Finally, the data is averaged per day, so we end up with the average fraction of both the city and the airplane arrivals that are S/E/I/R over the course of a day. The simulated data is returned as an instance of the class `SimData`

# Data Analysis and Visualisation

Running a simulation returns an instance of the class `SimData` which is a wrapper class around a numpy array, designed to facilitate data analysis. It is initialised with:
- `array`: the actual data. This is a 6 dimensional array, containing the number of people registered in each city, in each of municipal/arrival/departure data, in each group, in each compartment, at each point in time, for each simulation.
- `axis_order`: a list defining what each axis of the `array` corresponds to. The default ordering is `['cities', 'datatypes', 'groups', 'compartments', 'sims', 'times']`
- `values_present`: a dict which stores which values of each axis are present. For example, if an instance of `SimData` only has data for compartments 'E' and 'R' then we would have `simdata.values_present['compartments'] == ['E', 'R']`. If an axis has been summed over but the dimension has been kept. then the value for that axis would be `'n/a'`.

`SimData` comes with a few important functions:
- `__getitem__`: this function was the main reason `SimData` was written. It allows the class to be sliced by a dictionary which will form the `values_present` of the `SimData` returned by the slice. For axes (keys) which are not specified in the slicing dictionary, all values are kept. Calling `SimData.filter` also calls `SimData.__getitem__` under the hood.
- `daily_avg`: this function averages the data in `array` each day and returns a new `SimData` with one timepoint per day.
- I have also wrapped a few useful unary and binary numpy functions to work appropriately with instances of `SimData`, including `sum, mean, std, __add__, __truediv__, max, argmax'`.

To visualise simulations, I have implemented a number of plotting functions and helper functions. Some of these helper functions are also implemented as class functions of the class `Travel`, but using them there is not advised, with exception of `Travel.plot_sims`. Some useful functions include:
- `plot_avg_vals`: takes in a dict of `SimData` and a list of filters. For each `SimData` and each filter, calculates the number of people at each timestep in that subset of the `SimData` and plots the average. Both mean/std and median/IQR are supported. Both time and total worldwide infections are supported for the x axis.
- `different_thresholds_diffs`: takes in a `SimData` and a pair of filters. It returns the difference between the time to detection for filter 0 and threshold = $A$ (the day on which the fraction of people in `SimData[filters[0]]` that are shedding/infected crosses some threshold fraction $A$, a number between 0 and 1), and the time to detection for filter 0 and threshold = $A$, for a range of values of $A$ and $B$.  The criteria for when someone counts as shedding/infected is expressed in `filters[i]['compartments']`, so setting this value to `['I']` means that people are only considered shedding in a relevant way when they are in the `I` compartment. 
-  `threshold_ratio_diffs` and `differences_vs_threshold` are both plots along some 1D slice of the 2D surface visualised in the output of `different_thresholds_diffs`.
    - In `threshold_ratio_diffs`, a pair of lists of thresholds are created from a pair of upper and lower limits, and a number of points to include in the list. (time detection for filter 0 and the $i$th threshold in the first list) - (time detection for filter 1 and the $i$th threshold in the second list) are plotted against the ratio of these two thresholds for all $i$.
    - In `differences_vs_threshold`, the thresholds are assumed to be the same for the two subgroups specified by the two filters, and the time difference is plotted against the value of this threshold. A dict of `SimData` is passed to this function and one curve is produced for each `SimData` in the dict.
- `differences_vs_variable` is similar to `differences_vs_threshold`. In this case, the dict of `SimData` should be data from a family of simulations that are run for a range of values of one parameter. Then, for each threshold given, the difference in time to detection for the two groups is plotted against the value of that parameter, assuming that both subpopulations specified by the two filters have the same threshold.