import logging
import pickle
import torch
import csv
import copy
import os
import shutil
from server import Server
import numpy as np
from collections import namedtuple

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from threading import Thread
# from DRL import PPO, Memory
from DRL import PPO


class PPOTrainServer(Server):
    """Federated learning server that uses ppo to train during selection."""

    # Run federated learning
    def run(self):
        rounds = self.config.fl.rounds
        episodes = self.config.fl.episodes
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports
        # self.flag = 0  # whether it is the first round (0:yes; 1:no)

        # firstly, all clients train on their own data
        self.init_step()
        # set up drl environment
        self.initial_env()
        # print(self.state.shape)
        logging.info('state shape: {}\n'.format(self.state.shape))
        ppo = PPO(self.state_dim, self.action_dim)

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        reward_records = []
        running_reward = -100
        accuracy_records =[]
        a_loss, v_loss = [], []

        for episode in range(1, episodes+1):
            logging.info('######## Episode {}/{} ########'.format(episode, episodes))
            self.reset()
            # print('initial weight:', self.state)
            score = 0
            accuracy = 0
            e_a_loss = 0
            e_v_loss = 0
            # logging.info('######## Score {} ########'.format(score))

            # Perform rounds of federated learning
            for round in range(1, rounds + 1):
                logging.info('**** Round {}/{} ****'.format(round, rounds))
                # Run the federated learning round

                # self.state = self.reduce_dimensionals()
                # print('new state:', self.state)
                s = self.get_state()
                # print('s:', s)
                self.action, self.action_prob = ppo.select_action(s)
                self.action = np.array(self.action).astype(int)

                logging.info('current action:{}'.format(self.action))
                accuracy, reward, done = self.step_train()

                next_state = self.get_state()
                # print('next state:', next_state)
                # ppo update every 30 rounds
                if ppo.store(self.Transition(s, self.action, self.action_prob, reward, next_state)):
                    action_loss, value_loss = ppo.update()
                    a_loss += action_loss
                    v_loss += value_loss
                score += reward
                # self.total_step += 1

                # Break loop when target accuracy is met
                if target_accuracy and (accuracy >= target_accuracy):
                    logging.info('Target accuracy reached.')
                    break

            # running_reward = running_reward * 0.9 + score * 0.1
            # running_reward = score / 30
            running_reward = score
            reward_records.append(running_reward)
            # action.append(runing_action)
            accuracy_records.append(accuracy)

            logging.info('| DRL Episode: {} | Average Reward: {}| \n'.format(episode, running_reward))

        path = str(self.config.model) + '_iid[' + str(self.config.data.IID) + ']_bias[' + \
               str(self.config.data.bias['primary']) + ']_worker[' + \
               str(self.config.clients.total) + ']_' + str(self.config.fl.episodes) + 'x' + \
               str(self.config.fl.rounds)
        ppo.save_param(path)

        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))

        csvFile1 = open('./save/ppo_reward_{}_iid[{}]_bias[{}]_worker[{}]_ep[{}x{}].csv'.
                        format(self.config.model, self.config.data.IID, self.config.data.bias['primary'], self.config.clients.total,
                               self.config.fl.episodes,
                               self.config.fl.rounds), 'w', newline='')
        writer1 = csv.writer(csvFile1)
        csvFile2 = open('./save/training_accuracy_{}_iid[{}]_bias[{}]_worker[{}]_ep[{}x{}].csv'.
                        format(self.config.model, self.config.data.IID, self.config.data.bias['primary'], self.config.clients.total,
                               self.config.fl.episodes,
                               self.config.fl.rounds), 'w', newline='')
        writer2 = csv.writer(csvFile2)
        csvFile3 = open('./save/a_loss_{}_iid[{}]_bias[{}]_worker[{}]_ep[{}x{}].csv'.
                        format(self.config.model, self.config.data.IID, self.config.data.bias['primary'],
                               self.config.clients.total,
                               self.config.fl.episodes,
                               self.config.fl.rounds), 'w', newline='')
        writer3 = csv.writer(csvFile3)
        csvFile4 = open('./save/v_loss_{}_iid[{}]_bias[{}]_worker[{}]_ep[{}x{}].csv'.
                        format(self.config.model, self.config.data.IID, self.config.data.bias['primary'],
                               self.config.clients.total,
                               self.config.fl.episodes,
                               self.config.fl.rounds), 'w', newline='')
        writer4 = csv.writer(csvFile4)
        writer1.writerow(reward_records)
        writer2.writerow(accuracy_records)
        writer3.writerow(a_loss)
        writer4.writerow(v_loss)
        csvFile1.close()
        csvFile2.close()
        csvFile3.close()
        csvFile4.close()

        plt.figure(num=1, figsize=(8, 5), )
        plt.plot([x for x in range(len(accuracy_records))], [y for y in accuracy_records])
        plt.title('DRL-based Federated learning')
        plt.xlabel('Episode')
        plt.ylabel(' Training Accuracy')
        plt.savefig('./save/fl_{}_iid[{}]_bias[{}]_worker[{}]_ep[{}x{}]_train_accuracy.png'.
                    format(self.config.model, self.config.data.IID, self.config.data.bias['primary'], self.config.clients.total,
                           self.config.fl.episodes,
                           self.config.fl.rounds))
        # plt.figure(num=2, figsize=(8, 5), )
        # plt.plot([x for x in range(len(a_loss))], [y for y in a_loss])
        # plt.title('DRL')
        # plt.xlabel('Episode')
        # plt.ylabel(' Training Loss')
        # plt.savefig('./save/drl_aloss_{}_iid[{}]_bias[{}]_worker[{}]_ep[{}x{}]_train_reward.png'.
        #             format(self.config.model, self.config.data.IID, self.config.data.bias['primary'],
        #                    self.config.clients.total,
        #                    self.config.fl.episodes,
        #                    self.config.fl.rounds))
        plt.figure(num=3, figsize=(8, 5), )
        plt.plot([x for x in range(len(v_loss))], [y for y in v_loss])
        plt.title('DRL')
        plt.xlabel('Episode')
        plt.ylabel(' Training Loss')
        plt.savefig('./save/drl_vloss_{}_iid[{}]_bias[{}]_worker[{}]_ep[{}x{}].png'.
                    format(self.config.model, self.config.data.IID, self.config.data.bias['primary'],
                           self.config.clients.total,
                           self.config.fl.episodes,
                           self.config.fl.rounds))
        plt.figure(num=4, figsize=(8, 5), )
        plt.plot([x for x in range(len(reward_records))], [y for y in reward_records])
        plt.title('DRL')
        plt.xlabel('Episode')
        plt.ylabel(' Training Reward')
        plt.savefig('./save/drl_{}_iid[{}]_bias[{}]_worker[{}]_ep[{}x{}]_train_reward.png'.
                    format(self.config.model, self.config.data.IID, self.config.data.bias['primary'],
                           self.config.clients.total,
                           self.config.fl.episodes,
                           self.config.fl.rounds))
        plt.show()

        # Continue federated learning
        # super().run()
    def init_step(self):
        self.step = 0
        self.action = np.zeros(self.config.clients.total, 'int32')  # action
        self.action_dim = self.action.shape[0]

        self.global_weights = self.model.state_dict()  # will be changed with self.global_weight
        self.local_weights = [self.global_weights for _ in range(self.config.clients.total)]

        # Begin distribution to all the clients
        self.action = [self.config.fl.batch_size for _ in range(self.config.clients.total)]
        logging.info('initial action:{}'.format(self.action))
        accuracy, reward, done = self.step_train()

        # 开始训练pca
        self.PCA_train()


    def initial_env(self):
        self.render = False
        self.Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])

        # self.action = np.zeros(self.config.clients.total, 'int32')  # action
        # self.global_weights = self.model.state_dict()   # will be changed with self.global_weight
        # self.local_weights = [self.global_weights for _ in range(self.config.clients.total)]
        self.state = self.get_state()  # state (will be changed with self.global_weight)
        # self.update_timestep = 1000
        # self.step = 0
        # self.total_step = 0

        # self.action_dim = self.action.shape[0]

        # fitting the PCA only once
        # logging.info('initial state:{}'.format(self.state))
        # self.PCA_train()

        # self.state = self.reduce_dimensionals()
        # self.state_dim = len(self.state)
        self.state_dim = self.state.shape[1]
        logging.info('state_dim:{}'.format(self.state_dim))

        # Begin distribution to all the clients
        # self.action = [5 for i in range(self.config.clients.total)]
        # logging.info('initial action:{}'.format(self.action))
        # accuracy, reward, done = self.step_train()

    # Flatten weights
    def flatten_weights(self, weights):
        weight_vecs = []
        if type(weights) != list:
            for _, weight in weights.items():
                weight_vecs.extend(weight.flatten())
        else:
            for _, weight in weights:
                weight_vecs.extend(weight.flatten())
        return weight_vecs

    def PCA_train(self):
        logging.info('Flattening weights...')
        weight_vecs = [self.flatten_weights(weight) for weight in self.local_weights]

        self.pca = PCA(n_components=2)
        # self.pca.fit(self.state.reshape(-1, 1))
        self.pca.fit(weight_vecs)
        logging.info('PCA training done...')

    def reduce_dimensionals(self, X):
        return self.pca.transform(X)

    def flatten_weight(self, w):
        param = {}
        p = []

        for key, value in w.items():
            param[key] = value.detach().cpu().numpy()
        for name, l in param.items():
            temp = np.array(l).ravel()
            p = np.append(p, temp)
        return p

    # 先reduce dimension，再get state
    def get_state(self):
        # 当前[w^t, w_1^t, ..., w_K^t]
        combined_weights = [self.flatten_weights(weight) for weight in self.local_weights]
        combined_weights.append(self.flatten_weights(self.global_weights))

        return self.reduce_dimensionals(combined_weights).reshape(1,-1)
        # param = {}
        # p = []
        #
        # for key, value in self.global_weights.items():
        #     param[key] = value.detach().cpu().numpy()
        # for name, l in param.items():
        #     temp = np.array(l).ravel()
        #     p = np.append(p, temp)
        #
        # param_local = {}
        # for client_weight in self.local_weights:
        #     for key, value in client_weight.items():
        #         param_local[key] = value.detach().cpu().numpy()
        #     for name, l in param_local.items():
        #         temp_local = np.array(l).ravel()
        #         p = np.append(p, temp_local)
        # # updates = []
        # # for weight in weights:
        # #     update = []
        # #     for i, (name, weight) in enumerate(weight):
        # #         bl_name, baseline = baseline_weights[i]
        # #
        # #         # Ensure correct weight is being updated
        # #         assert name == bl_name
        # #
        # #         # Calculate update
        # #         delta = weight - baseline
        #
        # return p

    def load_ini_model(self):
        shutil.copyfile(self.config.paths.model+'/ini_global.pth', self.config.paths.model+'/global.pth')
        logging.info('Load initial global model: {}'.format(self.config.paths.model+'/ini_global.pth'))

    def reset(self):
        # delete global model
        os.remove(self.config.paths.model + '/global.pth')

        self.action = np.zeros(self.config.clients.total, 'int32')  # action

        self.load_ini_model()   # this line is important

    # Federated learning phases
    # run in self.round() (server.py)
    def selection(self):
        import fl_model  # pylint: disable=import-error

        # clients = self.clients
        action = self.action

        sample_clients_index = [i for i in range(len(action)) if action[i] > 0]
        sample_clients = [self.clients[d] for d in sample_clients_index]

        return sample_clients

    def step_train(self):
        done = False
        # def round(self):
        # if sum(self.action):
        import fl_model  # pylint: disable=import-error

        # if self.flag == 0:
        #     sample_clients = [self.clients[d] for d in self.config.clients.total]
        #
        #     # Configure selected clients for different batch size
        #     for client in sample_clients:
        #         client.batch_size = self.config.fl.batch_size
        #
        #     # Configure sample clients
        #     self.configuration(sample_clients)

        # Select clients to participate in the round
        sample_clients = self.selection()

        logging.info("type of action:{}\n".format(type(self.action)))
        # Configure selected clients for different batch size
        if type(self.action)==list:
            for client in sample_clients:
                client.batch_size = self.action[client.client_id]  # .item()
        else:
            for client in sample_clients:
                client.batch_size = self.action[client.client_id].item()  # .item()

        # Configure sample clients
        self.configuration(sample_clients)

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client updates
        reports = self.reporting(sample_clients)

        # local weight (length = the number of sampled clients)
        weights = [(report.client_id, report.weights) for report in reports]
        # print('local weight:', len(weights))
        # print('client 1:', weights[0], type(weights[0]))
        # flatten the local_weight
        for weight in weights:
            # for index, (name, data) in enumerate(weight[1]):
            # print('client_id:', weight[0])
            # print('weight:', weight[1])
            self.local_weights[weight[0]] = weight[1]
        # self.local_weights = weights

        # Perform weight aggregation
        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.save_reports(round, reports)

        # Save updated global model
        self.save_model(self.model, self.config.paths.model)

        # Test global model accuracy
        if self.config.clients.do_test:  # Get average accuracy from client reports
            accuracy = self.accuracy_averaging(reports)
        else:  # Test updated model on server
            testset = self.loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)

        if self.config.fl.target_accuracy and (accuracy >= self.config.fl.target_accuracy):
            logging.info('Target accuracy reached.')
            done = True

        reward = pow(64, (accuracy - self.config.fl.target_accuracy)) - 1
        self.step += 1
        logging.info('-| DRL Step : {} | Reward: {}| \t Step_accuracy:{:.2f}%\n'.format(self.step, reward, 100 * accuracy))

        return accuracy, reward, done
