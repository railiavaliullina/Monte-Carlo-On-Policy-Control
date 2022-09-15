import gym
import numpy as np
import time
from copy import deepcopy

from utils.visualization import Visualization
from utils.data_utils import read_file


class MC(object):
    """
    Class with first/every visit policy evaluation implementation.
    """

    def __init__(self, cfg):
        """
        Initializes class.
        :param cfg: config
        """
        self.cfg = cfg
        self.visualization = Visualization(cfg)

        self.init_env()
        self.show_env_info()

        self.transition_matrix = self.env.transition_matrix
        self.states_num = len(self.transition_matrix)
        self.actions_num = len(self.transition_matrix[0])
        self.actions_space = self.env.action_space.n
        self.get_policy()
        self.init_matrices()
        self.get_model_with_policy()

    def init_env(self):
        """
        Initializes environment with parameters from config.
        """
        self.env = gym.make(f'frozen_lake:{self.cfg.env_type}', map_name=self.cfg.map_name,
                            action_set_name=self.cfg.action_set)
        self.env.reset(start_state_index=0)

    def show_env_info(self):
        """
        Prints information about current environment.
        """
        if self.cfg.verbose:
            self.env.render(object_type="environment")
            self.env.render(object_type="actions")
            self.env.render(object_type="states")
            print(f'\nMap type: {self.cfg.map_name}\n'
                  f'Policy type: {self.cfg.policy_type}\n'
                  f'Action Set: {self.cfg.action_set}\n'
                  f'Discount Factor: {self.cfg.discount_factor}')

    def init_matrices(self):
        """
        init transition probability matrix, reward matrix.
        """
        self.transition_prob_matrix = np.zeros((self.states_num, self.states_num, self.actions_num))
        self.reward_matrix = np.zeros((self.states_num, self.actions_num))

        for s in range(self.states_num):
            for a in range(self.actions_num):
                for new_s_tuple in self.transition_matrix[s][a]:
                    transition_prob, new_s, reward, _ = new_s_tuple
                    self.transition_prob_matrix[s][new_s][a] += transition_prob
                    self.reward_matrix[s][a] += reward * transition_prob

    def get_model_with_policy(self):
        """
        Gets transition probability matrix, reward matrix within chosen policy.
        :return:
        """
        self.transition_prob_matrix_pi = np.zeros((self.states_num, self.states_num))
        self.reward_matrix_pi = np.zeros(self.states_num)

        for s in range(self.states_num):
            for new_s in range(self.states_num):
                self.transition_prob_matrix_pi[s][new_s] = self.policy[s] @ self.transition_prob_matrix[s][new_s]
            self.reward_matrix_pi[s] = self.policy[s] @ self.reward_matrix[s]

    def get_policy(self):
        """
        Initializes policy within config params.
        """
        self.policy = np.random.uniform(0, 1, (self.states_num, self.actions_num))
        self.policy = [self.policy[i, :] / np.sum(self.policy, 1)[i] for i in range(self.states_num)]

    def get_optimal_policy(self):
        data = read_file(self.cfg.optimal_policies_file_path)
        self.optimal_policy = np.asarray(data[self.cfg.map_name][self.cfg.action_set])

    def generate_episode(self, policy, on_policy=False):

        s = 0 if on_policy else np.random.choice(self.states_num)
        self.env.reset(start_state_index=s)

        if on_policy:
            episode = []
        else:
            action = np.random.choice(self.actions_num)
            while policy[s][action] == 0:
                action = np.random.choice(self.actions_num)
            s, reward, done, _ = self.env.step(action)
            episode = [(s, reward, action)]

        while True:
            action = np.random.choice(np.arange(self.actions_num), 1, p=policy[s])[0]
            new_state_index, reward, done, _ = self.env.step(action)
            episode.append((s, reward, action))
            if done:
                break
            s = new_state_index
        return np.asarray(episode)

    def mc_control_with_exploring_starts(self):
        """
        Runs exploring starts algorithm.
        :return: policy accuracy comparing to optimal policy.
        """
        Pi = np.ones((self.states_num, self.actions_num)) / 4
        Q = np.zeros((self.states_num, self.actions_num))
        returns = [[[] for _ in range(self.actions_num)] for _ in range(self.states_num)]
        step = 0
        accuracies, mean_accuracies = [], []

        while step < self.cfg.num_episodes:

            if step % 100 == 0:
                acc = np.mean(np.argmax(Pi, 1) == self.optimal_policy)
                accuracies.append(acc)
                mean_acc = np.mean(accuracies[:-10]) if len(accuracies) > 10 else np.mean(accuracies)
                print(f'step: {step}/{self.cfg.num_episodes}, accuracy: {mean_acc}')
                mean_accuracies.append(mean_acc)

            episode = self.generate_episode(Pi)

            S, R, A = episode[:, 0], episode[:, 1], episode[:, 2]
            S, A = S.astype(int), A.astype(int)
            T = len(episode)
            G = 0
            state_action_pairs = [(s, a) for s, a in zip(S, A)]
            # state_action_pairs = []

            for t in np.arange(T - 1, -1, -1):
                G = self.cfg.discount_factor * G + R[t]

                if (S[t], A[t]) not in state_action_pairs[:t]:
                    returns[S[t]][A[t]].append(G)
                    Q[S[t]][A[t]] = np.mean(returns[S[t]][A[t]])
                    # Pi[S[t]] = np.argmax(Q[S[t]])
                    Pi[S[t]] = np.zeros(len(Pi[S[t]]))
                    Pi[S[t]][np.argmax(Q[S[t]])] = 1
                    # state_action_pairs.append((S[t], A[t]))
            step += 1

        return mean_accuracies

    def mc_control_on_policy(self):
        Pi = np.ones((self.states_num, self.actions_num)) / 4
        Q = np.zeros((self.states_num, self.actions_num))
        returns = [[[] for _ in range(self.actions_num)] for _ in range(self.states_num)]
        step = 0
        accuracies, mean_accuracies = [], []

        while step < self.cfg.num_episodes:

            if step % 100 == 0:
                acc = np.mean(np.argmax(Pi, 1) == self.optimal_policy)
                accuracies.append(acc)
                mean_acc = np.mean(accuracies[:-10]) if len(accuracies) > 10 else np.mean(accuracies)
                print(f'step: {step}/{self.cfg.num_episodes}, accuracy: {mean_acc}')
                mean_accuracies.append(mean_acc)

            episode = self.generate_episode(Pi, on_policy=True)
            S, R, A = episode[:, 0], episode[:, 1], episode[:, 2]
            S, A = S.astype(int), A.astype(int)
            T = len(episode)
            G = 0
            state_action_pairs = [(s, a) for s, a in zip(S, A)]

            for t in np.arange(T - 1, -1, -1):
                G = self.cfg.discount_factor * G + R[t]

                if (S[t], A[t]) not in state_action_pairs[:t]:
                    returns[S[t]][A[t]].append(G)
                    Q[S[t]][A[t]] = np.mean(returns[S[t]][A[t]])

                    Pi[S[t]] = self.cfg.eps / self.actions_num  # np.zeros(len(Pi[S[t]]))
                    a_ = np.argmax(Q[S[t]])
                    Pi[S[t]][a_] = 1 - self.cfg.eps + Pi[S[t]][a_]

            step += 1

        return mean_accuracies

    def run(self):
        """
        Runs exploring starts/on policy control algorithms.
        """

        self.get_optimal_policy()
        print('\nExploring starts...\n')
        acc_exploring_starts = self.mc_control_with_exploring_starts()

        print('\nOn policy control...\n')
        acc_on_policy_control = self.mc_control_on_policy()

        return np.asarray(acc_exploring_starts), np.asarray(acc_on_policy_control)
