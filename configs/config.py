from easydict import EasyDict

from enums.enums import *

cfg = EasyDict()

cfg.optimal_policies_file_path = '../data/optimal_policies.json'
cfg.results_file_path = '../data/'

cfg.map_name = MapName.small.name  # изменить map_name: MapName.small.name, MapName.medium.name etc.
cfg.action_set = ActionSet.default.name  # изменить action set: ActionSet.default.name, ActionSet.slippery.name

cfg.policy_type = PolicyType.stochastic.name  # изменить policy_type: PolicyType.optimal.name, PolicyType.stochastic.name
cfg.env_type = EnvType.fall.value

cfg.discount_factor = 0.1
cfg.num_episodes = 1000
cfg.eps = 0.1

cfg.run_single_exp = True
cfg.verbose = True
cfg.runs_num = 1000
cfg.plots_dir = '../plots/'
cfg.save_plots = True
