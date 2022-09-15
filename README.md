# Monte Carlo On Policy Control, Reinforcement Learning



Реализованы алгоритмы контроля:

- Monte Carlo with Exploring Starts;

- Monte Carlo On policy control.


1) файл для запуска:

`/executor/executor.py`

`action_set`, `map_name`, `policy_type` можно поменять в `/configs/config.py` 
через параметры `cfg.action_set`, `cfg.map_name`, `cfg.policy_type`.

2) реализация first/every policy evaluation:

`/MC/MC.py`

3) Графики сравнения алгоритмов по bias и variance:
   
   `/plots/`

Также при запуске `/executor/executor.py` графики построятся в браузере, где можно будет посмотреть все значения подробнее и 
значения bias и variance для first и every visit policy evaluation выведятся в консоль.
