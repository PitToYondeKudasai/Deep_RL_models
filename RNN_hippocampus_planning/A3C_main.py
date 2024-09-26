import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from collections import namedtuple
import numpy as np
import scipy.signal
import os, time

from net import ACModel_small
from env import Maze

Rollout = namedtuple('Rollout', 
                     ('action', 'reward', 'next_state', 'goal_state', 'pred_next_state', 'pred_goal_state', 'policy', 'value'))

class Params():
    def __init__(self):
        self.lr = 1e-3
        self.gamma = 1
        self.lmbda = 1
        self.device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mem_size = 256
        self.entropy_beta = 0.05
        self.predictive_beta = 0.5
        self.value_beta = 0.05
        self.actor_beta = 1.0
        self.num_processes = 6
        self.n_cols = 4
        self.n_rows = 4
        self.env_size = self.n_cols*self.n_rows
        self.n_act = 5
        self.depth_planning = 8
        self.delta_time = 400
        self.delta_time_planning = 120
        self.tot_time = 20_000 #ms
        self.env_name = f'Maze_{self.n_cols}x{self.n_rows}'
        self.input_dim = self.env_size + self.n_act + 2 + 2*self.env_size + (self.n_act-1) * self.depth_planning + 1
        self.n_training_episodes = 50_000
        self.batch_size = 40
        self.activation  = nn.Tanh()
        self.load = True
        self.save = True
        self.no_planning = True

def prepare_state(obs, prev_action: list, prev_rew: int, time_elapsed: float, walls, rollout, rew_signal, params):
    obs_t = F.one_hot(torch.tensor(obs), params.env_size).to(params.device)
    prev_action_t = torch.FloatTensor(prev_action).to(params.device)
    prev_rew_t = torch.FloatTensor([prev_rew]).to(params.device)
    time_elapsed_t = torch.FloatTensor([time_elapsed]).to(params.device)
    walls_t = torch.FloatTensor(walls).to(params.device)
    rollout_t = torch.FloatTensor(rollout.flatten()).to(params.device)
    rew_signal_t = torch.FloatTensor([rew_signal]).to(params.device)
    state = torch.cat([obs_t, prev_action_t, prev_rew_t, time_elapsed_t, walls_t, rollout_t, rew_signal_t])
    return state.unsqueeze(0).unsqueeze(0)

def discount_cumsum(x: np.array, discount: float):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def compute_loss(buffer: Rollout, params: Params):
    batch = Rollout(*zip(*buffer))
    vs = torch.cat(batch.value).squeeze()
    rews = np.array(batch.reward)
    pis = torch.cat(batch.policy)
    acts = np.stack(batch.action)
    td_target = torch.FloatTensor(discount_cumsum(rews, params.gamma).copy()).to(params.device)
    advantage = td_target - vs
    advantage = advantage.detach()
    advantage = advantage.squeeze()
    pi_a = pis.gather(1, torch.tensor(acts, dtype=torch.int64).to(params.device)).squeeze() 

    # Actor Loss
    actor_loss = -(torch.log(pi_a) * advantage).mean()

    # Critic Loss
    critic_loss = params.value_beta * F.smooth_l1_loss(vs.squeeze(), td_target.detach().squeeze())  
    
    # Entropy loss
    entropy = (torch.log(pis) * pis).sum(dim=1).mean()
    entropy_loss = params.entropy_beta * entropy  

    # World model loss
    world_model_loss = -params.predictive_beta * ((torch.cat(batch.next_state).squeeze() * torch.log(torch.cat(batch.pred_next_state))) + 
                                                  (torch.stack(batch.goal_state) * torch.log(torch.cat(batch.pred_goal_state)))).mean()

    #print(entropy_loss, critic_loss, actor_loss, world_model_loss)
    loss = entropy_loss + critic_loss + actor_loss + world_model_loss
    return loss/params.batch_size

def get_simulation(model: nn.Module, obs, goal_state, h_in: torch.Tensor, c_in: torch.Tensor, time_elapsed: float, walls, params, max : bool=False):
    h_sim = torch.clone(h_in).detach()
    c_sim = torch.clone(c_in).detach()
    counter = 0
    rew_signal = 0 
    last_action = [0,0,0,0,1]
    goal_state = goal_state.argmax().item()
    planning_rollout = np.zeros([params.depth_planning, params.n_act-1])
    while (counter < params.depth_planning) and rew_signal==0:
        state = prepare_state(obs=obs, 
                              prev_action=last_action, 
                              prev_rew=0, 
                              time_elapsed=time_elapsed, 
                              walls=walls,
                              rollout=np.zeros([params.depth_planning, params.n_act-1]),
                              rew_signal=0,  
                              params=params)
        with torch.no_grad():
            _, _, action, pred_next_state, _, h_sim, c_sim = model(state, h_sim, c_sim, best_action=max, simulation=True)
        obs = pred_next_state.argmax().item()
        planning_rollout[counter, action] = 1
        if obs == goal_state:
            rew_signal = 1
            break
        last_action = np.eye(params.n_act)[action]
        counter += 1
    return planning_rollout, rew_signal

def get_trajectory(model: nn.Module, env: Maze, params: Params):
    buffer = []
    time_elapsed = 0
    h_in = torch.zeros([1, 1, params.mem_size]).to(params.device)
    c_in = torch.zeros([1, 1, params.mem_size]).to(params.device)   
    done_episode = False
    obs, rew, done_trial, _ = env.reset(reset_maze=True, reset_agent_pos=True, reset_goal_pos=True)
    goal_state = F.one_hot(torch.tensor(env.goal_position), params.env_size).to(params.device)
    state = prepare_state(obs=obs, 
                          prev_action=[0,0,0,0,0], 
                          prev_rew=rew, 
                          time_elapsed=time_elapsed/params.tot_time, 
                          walls=env.is_in_check, 
                          rollout=np.zeros([params.depth_planning, params.n_act-1]),
                          rew_signal=0,
                          params=params)
    tot_rew = 0
    tot_planning = 0
    n_steps = 0
    while not done_episode:
        n_steps += 1
        pi, v, action, pred_next_state, pred_goal_state, h_out, c_out = model(state, h_in, c_in, simulation=params.no_planning)
        if action == 4:
            time_elapsed += params.delta_time_planning
            planning_rollout, rew_signal = get_simulation(model, obs, pred_goal_state, h_out, c_out, 
                                                          time_elapsed/params.tot_time, 
                                                          env.is_in_check, params, max=False)
            rew = 0
            tot_planning += 1
        else:
            time_elapsed += params.delta_time
            obs, rew, done_trial, _, = env.step(action)
            planning_rollout = np.zeros([params.depth_planning, params.n_act-1])
            rew_signal = 0 
        action_one_hot = np.eye(params.n_act)[action]
        state = prepare_state(obs=obs, 
                              prev_action=action_one_hot, 
                              prev_rew=rew, 
                              time_elapsed=time_elapsed/params.tot_time, 
                              walls=env.is_in_check,
                              rollout=planning_rollout,
                              rew_signal=rew_signal,  
                              params=params)
        tot_rew += rew
        if time_elapsed >= params.tot_time:
            done_episode = True  
        buffer += [Rollout([action], rew, state[:, :, :params.env_size], goal_state, pred_next_state, pred_goal_state, pi, v)]
        h_in, c_in = h_out, c_out
        if done_trial:
            _, _, _, _, _, h_in, c_in = model(state, h_in, c_in)
            obs, _, done_trial, _ = env.reset(reset_maze=False, reset_agent_pos=True)
            state = prepare_state(obs=obs, 
                                  prev_action=action_one_hot, 
                                  prev_rew=rew, 
                                  time_elapsed=time_elapsed/params.tot_time, 
                                  walls=env.is_in_check, 
                                  rollout=np.zeros([params.depth_planning, params.n_act-1]),
                                  rew_signal=1,  
                                  params=params)      
    _ = model(state, h_in, c_in)
    return buffer, tot_rew, tot_planning/n_steps

def run(proc_name, global_model, global_optimizer, train_queue, res_queue, params):
    env = Maze(params.n_cols, params.n_rows, 40)
    all_rews = []
    all_planning_ratios = []
    for episode in range(params.n_training_episodes):
        global_model.zero_grad()
        buffer, tot_rew, planning_ratio = get_trajectory(global_model, env, params)
        all_rews.append(tot_rew)
        all_planning_ratios.append(planning_ratio)
        loss = compute_loss(buffer, params)
        global_optimizer.zero_grad()
        loss.backward()  
        if episode % 100 == 0:
            print(f'agent {proc_name} episode {episode} avg rew: {np.round(np.mean(all_rews[-100:]), 2)} planning % {np.round(np.mean(all_planning_ratios[-100:]), 2)}')  
        #clip_grad_norm_(global_model.parameters(), 10)
        grads = [
            param.grad.data.cpu().numpy()
            if param.grad is not None else None
            for param in global_model.parameters()
        ]
        train_queue.put(grads)
        res_queue.put((tot_rew, planning_ratio))
    train_queue.put(None)   
    res_queue.put(None)

if __name__ == "__main__":
    start = time.time()
    params = Params()
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    global_model = ACModel_small(params.input_dim, params.env_size, params.n_act, params.mem_size, params.activation).to(params.device)
    global_model.share_memory()
    global_optimizer = optim.Adam(global_model.parameters(), lr=params.lr)
    results = []
    path_res = 'results/'
    id_net = 256
    plan_or_not_plan = 'no_planning' if params.no_planning else 'planning'

    if params.load:
        checkpoint = torch.load(f'{path_res}checkpoint_{params.env_name}_{plan_or_not_plan}_{id_net}.pt')
        global_model.load_state_dict(checkpoint['model_state_dict'])
        global_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train_queue = mp.Queue(maxsize=params.batch_size)
    res_queue = mp.Queue(maxsize=params.batch_size)
    data_proc_list = []
    for rank in range(params.num_processes):
        data_proc = mp.Process(target=run, 
                               args=(rank, global_model, global_optimizer, train_queue, res_queue, params))
        data_proc.start()
        data_proc_list.append(data_proc)   
    batch = []
    step_idx = 0
    grad_buffer = None

    try:
        while True:
            train_entry = train_queue.get()
            results.append(res_queue.get())
            if train_entry is None:
                break

            step_idx += 1

            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for tgt_grad, grad in zip(grad_buffer, train_entry):
                    tgt_grad += grad

            if step_idx % params.batch_size == 0:
                for param, grad in zip(global_model.parameters(), grad_buffer):
                    param.grad = torch.FloatTensor(grad).to(params.device)

                #clip_grad_norm_(global_model.parameters(), 10)
                global_optimizer.step()
                grad_buffer = None
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()

    if params.save:
        torch.save({
            'model_state_dict': global_model.state_dict(),
            'optimizer_state_dict': global_optimizer.state_dict(),
            }, f'{path_res}checkpoint_{params.env_name}_{plan_or_not_plan}_{id_net}.pt')
    results = np.array(results[:-params.num_processes])
    if params.load:
        res = np.loadtxt(f'{path_res}res_{params.env_name}_{plan_or_not_plan}_{id_net}.txt')
        plan = np.loadtxt(f'{path_res}planning_{params.env_name}_{plan_or_not_plan}_{id_net}.txt')
        res = np.concatenate([res, results[:, 0]])
        plan = np.concatenate([plan, results[:, 1]])
    else:
        res = results[:, 0]
        plan = results[:, 1]
    np.savetxt(f'{path_res}res_{params.env_name}_{plan_or_not_plan}_{id_net}.txt', res)
    np.savetxt(f'{path_res}planning_{params.env_name}_{plan_or_not_plan}_{id_net}.txt', plan)
    print(time.time() - start) 