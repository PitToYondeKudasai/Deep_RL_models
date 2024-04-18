import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
import torch.optim as optim

from collections import namedtuple
import numpy as np
import os
import time

from net import ACModel
from env import Maze

Rollout = namedtuple('Rollout', ('state', 'action', 'reward', 'done', 'h_in', 'c_in'))

def train(global_model, optimizer, states_t, actions_t, rews_t, hs_t, cs_t, dones_t, params):
    state = states_t[:-1]
    next_state = states_t[1:]
    action =  actions_t
    reward = rews_t
    h_in = hs_t[0, :].unsqueeze(0)
    c_in = cs_t[0, :].unsqueeze(0)
    next_h_in = hs_t[1, :].unsqueeze(0)
    next_c_in = cs_t[1, :].unsqueeze(0)
    done = dones_t
    done_mask = torch.FloatTensor(1 - np.array(done)).to(params.device)

    with torch.no_grad():
        _, v_prime, _, _ = global_model(next_state, next_h_in, next_c_in)
    td_target = (torch.FloatTensor(reward).to(params.device) + params.gamma * v_prime.squeeze() * done_mask.squeeze())
    pi, v, _, _  = global_model(state, h_in, c_in) 
    delta = td_target - v.squeeze()
    delta = delta.detach().cpu().numpy()
    advantage_lst = []
    advantage = 0.0
    for item in delta[::-1]:
        advantage = params.gamma * params.lmbda * advantage + item
        advantage_lst.append([advantage])
    advantage_lst.reverse()
    advantage = torch.FloatTensor(np.array(advantage_lst)).squeeze().to(params.device)
    act = torch.tensor(action, dtype=torch.int64).to(params.device).unsqueeze(-1)
    pi_a = pi.gather(2, act).squeeze()
    # Actor Loss
    actor_loss = -(torch.log(pi_a) * advantage).mean()

    # Critic Loss
    critic_loss = F.smooth_l1_loss(v.squeeze(), td_target.detach().squeeze())

    # Entropy loss
    entropy = (torch.log(pi) * pi).sum(dim=1).mean()
    entropy_loss = params.entropy_beta * entropy

    loss = entropy_loss + critic_loss + actor_loss

    optimizer.zero_grad()
    loss.mean().backward(retain_graph=False)
    optimizer.step()


def prepare_state(obs, prev_action, prev_rew, time_elapsed, walls, params):
    obs_t = F.one_hot(torch.tensor(obs), params.n_cols * params.n_rows).to(params.device)
    prev_action_t = torch.FloatTensor(prev_action).to(params.device)
    prev_rew_t = torch.FloatTensor([prev_rew]).to(params.device)
    time_elapsed_t = torch.FloatTensor([time_elapsed]).to(params.device)
    walls_t = torch.FloatTensor(walls).to(params.device)
    state = torch.cat([obs_t, prev_action_t, prev_rew_t, time_elapsed_t, walls_t])
    return state.unsqueeze(0).unsqueeze(0)

def run(global_model, params, train_queue, rank):
    # We create local model
    local_model = ACModel(params.input_dim, params.n_act, params.mem_size, nn.Tanh()).to(params.device)
    # We create an environment
    env = Maze(params.n_cols, params.n_rows, 40)
    all_rews = []
    for episode in range(params.n_training_episodes):
        local_model.load_state_dict(global_model.state_dict())
        # We create the buffer
        buffer = []
    
        time_elapsed = 0
        h_in = torch.zeros([1, 1, params.mem_size]).to(params.device)
        c_in = torch.zeros([1, 1, params.mem_size]).to(params.device)
        done_episode = False
        obs, rew, done_trial, _ = env.reset(reset_maze=True, reset_agent_goal_pos=True)
        state = prepare_state(obs=obs, 
                            prev_action=[0,0,0,0], 
                            prev_rew=rew, 
                            time_elapsed=time_elapsed, 
                            walls=env.is_in_check, 
                            params=params)
        tot_rew = 0
        while not done_episode:
            with torch.no_grad():
                pi, v, h_out, c_out = local_model(state, h_in, c_in)
            prob = Categorical(pi.squeeze())
            action = prob.sample().item()
            new_obs, rew, done_trial, _, = env.step(action)
            time_elapsed += params.delta_time
            action_one_hot = np.eye(4)[action]
            new_state = prepare_state(obs=new_obs, 
                                      prev_action=action_one_hot, 
                                      prev_rew=rew, 
                                      time_elapsed=time_elapsed, 
                                      walls=env.is_in_check, 
                                      params=params)
            tot_rew += rew
            if time_elapsed >= 1:
                done_episode = True
            # Save in the buffer
            action = np.array([action])
            buffer += [Rollout(state, action, [rew], [done_episode], h_in, c_in)]
            h_in, c_in = h_out, c_out
            state = new_state
            if done_episode:
                buffer += [Rollout(state, None, None, None, None, None)]
            if done_trial:
                new_obs, _, done_trial, _ = env.reset(reset_maze=False)
                state = prepare_state(obs=new_obs, 
                                      prev_action=action_one_hot, 
                                      prev_rew=rew, 
                                      time_elapsed=time_elapsed, 
                                      walls=env.is_in_check, 
                                      params=params)
        batch = Rollout(*zip(*buffer))
        all_rews.append(tot_rew)
        if episode % 10 == 0:
            print(f'agent {rank} episode {episode} avg rew: {np.round(np.mean(all_rews[-10:]), 2)}')
        train_queue.put(batch)
    train_queue.put('Done')

class Params():
    def __init__(self):
        self.lr = 2e-4 # 2e-4
        self.gamma = 0.95
        self.lmbda = 0.95
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mem_size = 128
        self.entropy_beta = 0.05
        self.seed = 1
        self.num_processes = 5
        self.n_cols = 4
        self.n_rows = 4
        self.n_act = 4
        self.delta_time = 0.01
        self.env_name = f'Maze_{self.n_cols}x{self.n_rows}'
        self.input_dim = self.n_cols*self.n_rows + self.n_act + 2 + 32
        self.n_training_episodes = 20_000
        self.batch_size = 5
        self.load = True
        self.save = True


if __name__ == "__main__":
    start = time.time()
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    params = Params()
    global_model = ACModel(params.input_dim, params.n_act, params.mem_size, nn.Tanh()).to(params.device)
    optimizer = optim.Adam(global_model.parameters(), lr=params.lr)

    if params.load:
        checkpoint = torch.load('checkpoint_1.pt')
        global_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    PROCESSES_COUNT = params.num_processes
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for rank in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=run, args=(global_model, params, train_queue, rank))
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = {
        'states': [],
        'actions': [],
        'rewards': [],
        'hs': [],
        'cs': [],
        'dones': []
    }

    try:
        while True:
            train_input = train_queue.get()
            if isinstance(train_input, str):
                break
            # Try to stack the 5 episodes in a batch before trainig 
            batch['states'].append(torch.cat(train_input.state))
            batch['actions'].append(np.stack(train_input.action[:-1]))
            batch['rewards'].append(np.stack(train_input.reward[:-1]))
            batch['hs'].append(torch.cat(train_input.h_in[:2]))
            batch['cs'].append(torch.cat(train_input.c_in[:2]))
            batch['dones'].append(np.stack(train_input.done[:-1]))
            if  len(batch['states']) == params.batch_size: 
                states_t = torch.cat(batch['states'], 1)
                actions_t = np.concatenate(batch['actions'], 1)
                rews_t = np.concatenate(batch['rewards'], 1)
                hs_t = torch.cat(batch['hs'], 1)
                cs_t = torch.cat(batch['cs'], 1)
                dones_t = np.concatenate(batch['dones'], 1)
                train(global_model, optimizer, states_t, actions_t, rews_t, hs_t, cs_t, dones_t, params)
                for key in batch:
                    batch[key].clear()
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()

    if params.save:
        checkpoint = { 
            'model': global_model.state_dict(),
        }
        torch.save({
            'model_state_dict': global_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'checkpoint_1.pt')
    print(time.time() - start)