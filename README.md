# Monte Carlo On Policy Control, Reinforcement Learning


## About The Project

The following control algorithms were implemented:

- Monte Carlo with Exploring Starts;

- Monte Carlo On policy control.



## Getting Started

File to run: 
    
    /executor/executor.py

Parameters `action_set`, `map_name`, `policy_type` can be changed in:
    
    /configs/config.py via parameters cfg.action_set, cfg.map_name, cfg.policy_type.

The implementation of algorithms is in:

      /MC/MC.py

Visualization for comparing algorithms by bias and variance is in:
   
    /plots/

## Additional Information

Also, when you run `/executor/executor.py`, graphs will be built in the browser, where you can see all the values in more detail and the bias and variance values for first and every visit policy evaluation will be printed to the console.
