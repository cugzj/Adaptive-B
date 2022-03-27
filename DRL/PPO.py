import os, time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical, multinomial
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

gamma = 0.9
# action_spaces = [0,10,20,30,40,50,60,70,80,90,100]


class ActorNet(nn.Module):

    def __init__(self, num_state, num_action):
        super(ActorNet, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.fc1 = nn.Linear(self.num_state, 100)
        self.fc2 = nn.Linear(100, 100)
        # self.action_head = nn.Linear(100, num_action)
        self.mu_head = nn.Linear(100, self.num_action)
        self.sigma_head = nn.Linear(100, self.num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # action_prob = F.softmax(self.action_head(x), dim=1)
        # return action_prob
        mu = 50.0 * F.tanh(self.mu_head(x))
        sigma = 0.05 * F.softplus(self.sigma_head(x))
        # print('mu:{}'.format(mu))
        # print('sigma:{}'.format(sigma))
        return (mu, sigma)


class CriticNet(nn.Module):

    def __init__(self, num_state, num_action):
        super(CriticNet, self).__init__()
        self.num_state = num_state
        self.num_action = num_action
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, 100)
        self.v_head = nn.Linear(100, 1)
        # self.v_head = nn.Linear(100, self.num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_value = self.v_head(x)
        return state_value


class PPO():

    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity, batch_size = 10, 10   # update in each episode

    def __init__(self, num_state, num_action):
        self.training_step = 0
        self.num_state = num_state
        self.num_action = num_action
        self.anet = ActorNet(self.num_state,self.num_action).double()
        self.cnet = CriticNet(self.num_state,self.num_action).double()
        self.buffer = []
        self.counter = 0
        # self.writer = SummaryWriter('../exp')

        self.optimizer_a = optim.Adam(self.anet.parameters(), lr=1e-4)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=3e-4)

        self.policy_old = ActorNet(self.num_state, self.num_action).double()
        self.policy_old.load_state_dict(self.anet.state_dict())

        if not os.path.exists('./param'):
            os.makedirs('./param/net_param')
            os.makedirs('./param/img')
            # os.makedirs('./param/')

    def select_action(self, state):
        # def act(self, state):
        state = torch.from_numpy(state).double().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.anet(state)
        # action = np.random.choice(action_spaces, self.num_action, replace=True, p=action_prob)
        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = action.clamp(0, 100)
        # print('size of action:{} \t action:{}'.format(action.size(), action))
        # print('action_log_prob:{}'.format(action_log_prob[0].numpy()))
        return action[0].numpy(), action_log_prob[0].numpy()

    def get_value(self, state):

        state = torch.from_numpy(state).double().unsqueeze(0)
        with torch.no_grad():
            state_value = self.cnet(state)
        return state_value.item()

    def save_param(self, path):
        torch.save(self.anet.state_dict(), './param/ppo_anet_' + path + '.pth')
        torch.save(self.cnet.state_dict(), './param/ppo_cnet_' + path + '.pth')

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1

        s = torch.tensor([t.s for t in self.buffer], dtype=torch.double)
        # a = torch.tensor([t.a for t in self.buffer], dtype=torch.double).view(-1, 1)
        a = torch.tensor([t.a for t in self.buffer], dtype=torch.double)
        r = torch.tensor([t.r for t in self.buffer], dtype=torch.double).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in self.buffer], dtype=torch.double)

        old_action_log_probs = torch.tensor(
            # [t.a_log_p for t in self.buffer], dtype=torch.double).view(-1, 1)
            [t.a_log_p for t in self.buffer], dtype = torch.double)

        r = (r - r.mean()) / (r.std() + 1e-5)
        with torch.no_grad():
            target_v = r + gamma * self.cnet(s_)

        adv = (target_v - self.cnet(s)).detach()

        a_loss = []
        v_loss = []

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                (mu, sigma) = self.anet(s[index])
                dist = Normal(mu, sigma)
                action_log_probs = dist.log_prob(a[index])
                ratio = torch.exp(action_log_probs - old_action_log_probs[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()

                self.optimizer_a.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.anet.parameters(), self.max_grad_norm)
                self.optimizer_a.step()

                value_loss = F.smooth_l1_loss(self.cnet(s[index]), target_v[index])
                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.cnet.parameters(), self.max_grad_norm)
                self.optimizer_c.step()

                a_loss.append(action_loss.item())
                v_loss.append(value_loss.item())

        # Copy new weights into old policy:
        # self.policy_old.load_state_dict(self.anet.state_dict())
        del self.buffer[:]

        print('ppo update!')

        # return sum(a_loss)/len(a_loss), sum(v_loss)/len(v_loss)
        return a_loss, v_loss
