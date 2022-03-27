import logging
import pickle
import torch
import csv
import copy
from server import Server
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from threading import Thread
from DRL import PPO_discrete


class FavorAvgServer(Server):
    """Federated learning server that uses ppo to train during selection."""

    # Run federated learning
    def run(self):
        rounds = self.config.fl.rounds
        episodes = self.config.fl.episodes
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports

        # filename and directory to load drl model from
        filename = "ppo_anet_favor_" + str(self.config.model) + "_iid[" + str(self.config.data.IID) + \
                   "]_bias[" + str(self.config.data.bias['primary']) + \
                   "]_worker[" + str(self.config.clients.total) + "]_" + "60" \
                   + "x" + "30" + ".pth"
        directory = "./param/"

        # set up drl environment
        self.initial_env()
        # memory = Memory()
        ppo = PPO_discrete(self.state_dim, self.action_dim)
        ppo.actor_net.load_state_dict(torch.load(directory+filename))

        # self.reset()

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        # reward_records = []
        accuracy_records =[]
        action_records = []

        # Perform rounds of federated learning
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))
            # Run the federated learning round

            s = self.state
            self.action, self.action_prob = ppo.select_action(s, self.config.clients.per_round)
            self.action = np.array(self.action).astype(int)

            logging.info('current action:{}'.format(self.action))
            accuracy, reward, done = self.step_train()

            action_records.append(self.action)
            accuracy_records.append(accuracy)

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break

        csvFile2 = open('./save/test/favor_training_accuracy_{}_iid[{}]_bias[{}]_worker[{}]_ep[{}x{}].csv'.
                        format(self.config.model, self.config.data.IID,
                               self.config.data.bias['primary'],
                               self.config.clients.total,
                               self.config.fl.episodes,
                               self.config.fl.rounds), 'w', newline='')
        writer2 = csv.writer(csvFile2)
        writer2.writerow(accuracy_records)
        csvFile2.close()

        plt.figure(num=1, figsize=(8, 5), )
        plt.plot([x for x in range(len(accuracy_records))], [y for y in accuracy_records])
        plt.title('DRL-based Federated learning')
        plt.xlabel('Communication Round')
        plt.ylabel(' Training Accuracy')
        plt.savefig('./save/test/favor_fl_{}_iid[{}]_bias[{}]_worker[{}]_ep[{}x{}]_train_accuracy.png'.
                    format(self.config.model, self.config.data.IID,
                           self.config.data.bias['primary'],
                           self.config.clients.total,
                           self.config.fl.episodes,
                           self.config.fl.rounds))
        plt.show()

        if reports_path:
            reports_test_path = "./save/test/reports.pkl"
            with open(reports_test_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_test_path))

        # Continue federated learning
        # super().run()

    def initial_env(self):
        self.render = False
        self.Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])

        self.action = np.zeros(self.config.clients.total, 'int32')  # action
        self.global_weights = self.model.state_dict()  # will be changed with self.global_weight
        self.state = self.get_state()  # state (will be changed with self.global_weight)
        self.update_timestep = 1000
        self.step = 0
        # self.total_step = 0

        self.action_dim = self.action.shape[0]

        # fitting the PCA only once
        # logging.info('initial state:{}'.format(self.state))
        # self.PCA_train()

        # self.state = self.reduce_dimensionals()
        self.state_dim = len(self.state)
        logging.info('state_dim:{}\t action_dim:{}'.format(self.state_dim,self.action_dim))

    def get_state(self):
        param = {}
        p = []
        for key, value in self.global_weights.items():
            param[key] = value.detach().cpu().numpy()
        for name, l in param.items():
            temp = np.array(l).ravel()
            p = np.append(p, temp)
        return p

    def selection(self):
        import fl_model  # pylint: disable=import-error

        # clients = self.clients
        action = self.action

        # sample_clients_index = [i for i in range(len(action))]
        sample_clients = [self.clients[d] for d in action]

        return sample_clients

    def step_train(self):
        done = False
        # def round(self):
        # if sum(self.action):
        import fl_model  # pylint: disable=import-error

        # Select clients to participate in the round
        sample_clients = self.selection()

        # Configure selected clients for different batch size
        for client in sample_clients:
            client.batch_size = self.config.fl.batch_size  # .item()

        # Configure sample clients
        self.configuration(sample_clients)

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

        # Recieve client updates
        reports = self.reporting(sample_clients)

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
        logging.info(
            '-| DRL Step : {} | Reward: {}| \t Step_accuracy:{:.2f}%\n'.format(self.step, reward, 100 * accuracy))

        return accuracy, reward, done
