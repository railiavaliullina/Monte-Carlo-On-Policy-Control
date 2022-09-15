import pickle

import numpy as np
import pandas as pd

from MC.MC import MC
from configs.config import cfg
from enums.enums import *
from utils.visualization import Visualization


class Executor(object):
    def __init__(self):
        """
        Class for running policy evaluation algorithms with different params several times and analysing results.
        """
        self.cfg = cfg
        np.random.seed(0)
        self.seeds = np.random.choice(int(5e5), self.cfg.runs_num, replace=False)
        self.map_names = [MapName.small.name, MapName.huge.name]
        self.action_sets = [a.name for a in ActionSet]
        self.visualization = Visualization(cfg)
        self.mc = MC(cfg)

    def write_results(self, accs_exploring_starts, accs_on_policy_control):
        """
        Writes all experiments results to pickle file.
        :param accs_exploring_starts: accuracies for exploring_starts algorithm
        :param accs_on_policy_control: accuracies for on_policy algorithm
        """
        exploring_starts_df = pd.DataFrame()
        on_policy_df = pd.DataFrame()

        for m_id, map_name in enumerate(self.map_names):
            for a_id, action_set in enumerate(self.action_sets):
                exploring_starts_df[f'{map_name}_{action_set}'] = np.mean(np.stack(accs_exploring_starts[map_name][action_set]), 0)
                on_policy_df[f'{map_name}_{action_set}'] = np.mean(np.stack(accs_on_policy_control[map_name][action_set]), 0)

        exploring_starts_df.to_pickle('../data/exploring_starts_df.pickle')
        on_policy_df.to_pickle('../data/on_policy_control_df.pickle')
        # self.visualization.plot_scatters()

    def run_sequence_of_experiments(self):
        """
        Runs sequence of experiments with different params.
        """
        accs_exploring_starts = {map_name: {action_set: [] for action_set in self.action_sets}
                                 for map_name in self.map_names}
        accs_on_policy_control = {map_name: {action_set: [] for action_set in self.action_sets}
                                  for map_name in self.map_names}

        for m_id, map_name in enumerate(self.map_names):
            self.cfg.map_name = map_name

            for a_id, action_set in enumerate(self.action_sets):
                self.cfg.action_set = action_set

                for run_id in range(self.cfg.runs_num):
                    print(f'Run: {run_id}/{self.cfg.runs_num}')
                    np.random.seed(self.seeds[run_id])
                    self.mc = MC(self.cfg)

                    acc_exploring_starts, acc_on_policy_control = self.mc.run()
                    accs_exploring_starts[map_name][action_set].append(acc_exploring_starts)
                    accs_on_policy_control[map_name][action_set].append(acc_on_policy_control)

        self.write_results(accs_exploring_starts, accs_on_policy_control)

    def run(self):
        """
        Runs whole pipeline.
        """
        if self.cfg.run_single_exp:
            self.visualization.plot_scatters()
            self.mc.run()
        else:
            # self.visualization.plot_scatters()
            self.run_sequence_of_experiments()


if __name__ == '__main__':
    executor = Executor()
    executor.run()
