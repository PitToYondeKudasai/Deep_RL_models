import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from collections import namedtuple
import numpy as np
import os, time, math

from net import ACModel
from env import Maze

Rollout = namedtuple('Rollout', ('action', 'reward', 'done', 'policy', 'value'))

class Params():
    def __init__(self):
        self.lr = 5e-4 
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
        self.n_training_episodes = 40_000
        self.batch_size = 5
        self.load = False
        self.save = True

def prepare_state(obs, prev_action, prev_rew, time_elapsed, walls, params):
    obs_t = F.one_hot(torch.tensor(obs), params.n_cols * params.n_rows).to(params.device)
    prev_action_t = torch.FloatTensor(prev_action).to(params.device)
    prev_rew_t = torch.FloatTensor([prev_rew]).to(params.device)
    time_elapsed_t = torch.FloatTensor([time_elapsed]).to(params.device)
    walls_t = torch.FloatTensor(walls).to(params.device)
    state = torch.cat([obs_t, prev_action_t, prev_rew_t, time_elapsed_t, walls_t])
    return state.unsqueeze(0).unsqueeze(0)

def compute_loss(buffer: list, params: Params) -> torch.float:
    batch = Rollout(*zip(*buffer))
    vs_prime = torch.stack(batch.value[1:]).detach().squeeze()
    vs = torch.cat(batch.value[:-1]).squeeze()
    rews = np.array(batch.reward[:-1])
    pis = torch.cat(batch.policy[:-1])
    acts = np.stack(batch.action[:-1])
    done_mask = torch.FloatTensor(1-np.array(batch.done[:-1])).to(params.device)
    td_target = (torch.FloatTensor(rews) + params.gamma * vs_prime * done_mask.squeeze())
    delta = td_target - vs
    delta = delta.detach().cpu().numpy()
    advantage_lst = []
    advantage = 0.0
    for item in delta[::-1]:
        advantage = params.gamma * params.lmbda * advantage + item
        advantage_lst.append([advantage])
    advantage_lst.reverse()
    advantage = torch.FloatTensor(np.array(advantage_lst)).squeeze().to(params.device)
    pi_a = pis.gather(1, torch.tensor(acts, dtype=torch.int64).to(params.device)).squeeze() 

    # Actor Loss
    actor_loss = -(torch.log(pi_a) * advantage).mean()

    # Critic Loss
    critic_loss = F.smooth_l1_loss(vs.squeeze(), td_target.detach().squeeze())  
    
    # Entropy loss
    entropy = (torch.log(pis) * pis).sum(dim=1).mean()
    entropy_loss = params.entropy_beta * entropy  

    loss = entropy_loss + critic_loss + actor_loss
    return loss

def get_trajectory(model: torch.nn.Module, env: Maze, params: Params):
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
        pi, v, h_out, c_out = model(state, h_in, c_in)
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
        buffer += [Rollout([action], rew, done_episode, pi, v)]
        h_in, c_in = h_out, c_out
        state = new_state 
        if done_trial:
            new_obs, _, done_trial, _ = env.reset(reset_maze=False)
            state = prepare_state(obs=new_obs, 
                                  prev_action=action_one_hot, 
                                  prev_rew=rew, 
                                  time_elapsed=time_elapsed, 
                                  walls=env.is_in_check, 
                                  params=params)      
    _, _, _, _ = model(state, h_in, c_in)
    return buffer, tot_rew

def run(proc_name, global_model, global_optimizer, train_queue, params):
    #local_model = ACModel(params.input_dim, params.n_act, params.mem_size, nn.Tanh()).to(params.device)
    env = Maze(params.n_cols, params.n_rows, 40)
    all_rews = []
    for episode in range(params.n_training_episodes):
        global_model.zero_grad()
        buffer, tot_rew = get_trajectory(global_model, env, params)
        all_rews.append(tot_rew)
        loss = compute_loss(buffer, params)
        global_optimizer.zero_grad()
        loss.backward()  
        if episode % 10 == 0:
            print(f'agent {proc_name} episode {episode} avg rew: {np.round(np.mean(all_rews[-10:]), 2)}')  
        clip_grad_norm_(global_model.parameters(), 10)
        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in global_model.parameters()
        ]
        train_queue.put(grads)

    train_queue.put(None)      

if __name__ == "__main__":
    start = time.time()
    params = Params()
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    global_model = ACModel(params.input_dim, params.n_act, params.mem_size, nn.Tanh()).to(params.device)
    global_model.share_memory()
    global_optimizer = optim.Adam(global_model.parameters(), lr=params.lr)

    if params.load:
        checkpoint = torch.load('checkpoint_1.pt')
        global_model.load_state_dict(checkpoint['model_state_dict'])
        global_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train_queue = mp.Queue(maxsize=params.num_processes)
    data_proc_list = []
    for rank in range(params.num_processes):
        data_proc = mp.Process(target=run, 
                               args=(rank, global_model, global_optimizer, train_queue, params))
        data_proc.start()
        data_proc_list.append(data_proc)   
    batch = []
    step_idx = 0
    grad_buffer = None

    try:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break

            step_idx += 1

            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for tgt_grad, grad in zip(grad_buffer,
                                          train_entry):
                    tgt_grad += grad

            if step_idx % params.num_processes == 0:
                for param, grad in zip(global_model.parameters(), grad_buffer):
                    param.grad = torch.FloatTensor(grad).to(params.device)

                clip_grad_norm_(
                    global_model.parameters(), 10)
                global_optimizer.step()
                grad_buffer = None
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
            'optimizer_state_dict': global_optimizer.state_dict(),
            }, 'checkpoint_1.pt')
    print(time.time() - start)