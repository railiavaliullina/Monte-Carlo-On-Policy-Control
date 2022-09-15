import plotly.express as px
import numpy as np
import pandas as pd


class Visualization(object):
    def __init__(self, cfg):
        """
        Class for visualizing policy, value as action set; run time and convergence time comparison plots.
        :param cfg: config
        """
        self.cfg = cfg

    def show_plot(self, exploring_starts_df, on_policy_control_df, map_size):
        exploring_starts_df_small_def = exploring_starts_df[f'{map_size}_default'].to_numpy()
        exploring_starts_df_small_slip = exploring_starts_df[f'{map_size}_slippery'].to_numpy()
        exploring_starts_df['color'] = [f'{map_size}_default'] * len(exploring_starts_df_small_def)
        exploring_starts_df['step'] = np.arange(len(exploring_starts_df_small_def))

        on_policy_control_df_small_def = on_policy_control_df[f'{map_size}_default'].to_numpy()
        on_policy_control_df_small_slip = on_policy_control_df[f'{map_size}_slippery'].to_numpy()

        df = pd.DataFrame()
        df['acc'] = list(exploring_starts_df_small_def) + list(exploring_starts_df_small_slip) \
                    + list(on_policy_control_df_small_def) + list(on_policy_control_df_small_slip)
        df['color'] = ['exploring_starts_default'] * len(list(exploring_starts_df_small_def)) + \
                      ['exploring_starts_slippery'] * len(list(exploring_starts_df_small_slip)) + \
                      ['on_policy_control_default'] * len(list(on_policy_control_df_small_def)) + \
                      ['on_policy_control_slippery'] * len(list(on_policy_control_df_small_slip))

        df['step'] = list(np.arange(len(list(exploring_starts_df_small_def)))) + \
                     list(np.arange(len(list(exploring_starts_df_small_slip)))) + \
                     list(np.arange(len(list(on_policy_control_df_small_def)))) + \
                     list(np.arange(len(list(on_policy_control_df_small_slip))))

        fig = px.line(df, x='step', y='acc', color='color')
        fig.update_layout(title=f'{map_size} map')
        fig.show()

    def plot_scatters(self):
        """
        Plots bias/variance comparison plots.
        Plots will be visualized in browser with ability to zoom in and will be saved as files.
        """
        exploring_starts_df = pd.read_pickle('../data/exploring_starts_df.pickle')
        on_policy_control_df = pd.read_pickle('../data/on_policy_control_df.pickle')

        self.show_plot(exploring_starts_df, on_policy_control_df, map_size='small')
        self.show_plot(exploring_starts_df, on_policy_control_df, map_size='huge')

        exploring_starts_df_huge_def = exploring_starts_df['huge_default'].to_numpy()
        exploring_starts_df_huge_slip = exploring_starts_df['huge_slippery'].to_numpy()

        df1 = pd.DataFrame()
        df1['acc'] = list(exploring_starts_df_huge_def) + list(exploring_starts_df_huge_slip)
        df1['color'] = ['default'] * len(list(exploring_starts_df_huge_def)) + \
                       ['slippery'] * len(list(exploring_starts_df_huge_slip))
        df1['step'] = list(np.arange(len(list(exploring_starts_df_huge_def)))) + \
                      list(np.arange(len(list(exploring_starts_df_huge_slip))))
        fig = px.line(df1, x='step', y='acc', color='color')
        fig.update_layout(title=f'Huge map')
