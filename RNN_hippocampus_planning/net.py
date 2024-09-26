import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ACModel_small(nn.Module):
    def __init__(self, input_dim, env_size, action_dim, mem_out_size=128, activation=nn.ReLU()):
        super(ACModel_small, self).__init__() 
        self.env_size = env_size
        self.action_dim = action_dim
        
        self.memory = nn.LSTM(input_dim, mem_out_size)

        self.fc_pi = nn.Sequential(
            nn.Linear(mem_out_size, action_dim),
        )
        
        self.fc_v = nn.Sequential(
            nn.Linear(mem_out_size, 1),
        )

        self.fc_states = nn.Sequential(
            nn.Linear(mem_out_size + action_dim, 32),
            activation, 
            nn.Linear(32, 2*env_size)
        )
            
    def forward(self, *args, best_action=False, simulation=False):
        x = args[0] 
        h_in, c_in = args[-2:]
        
        out, (h_out, c_out) = self.memory(x, (h_in, c_in))
        
        v = self.fc_v(out)
        pi = self.fc_pi(out)
        if simulation:
            pi = F.softmax(pi[:, :, :-1], dim=2).squeeze(1)
        else:
            pi = F.softmax(pi, dim=2).squeeze(1)
            
        if not best_action:
            prob = Categorical(pi.squeeze())
            action = prob.sample().item()
        else:
            action = pi.argmax().item()
        action_one_hot = torch.eye(self.action_dim)[action]
        next_s_goal_s = self.fc_states(torch.cat([h_out.squeeze(0), action_one_hot.unsqueeze(0)], dim=1))
        next_state = F.softmax(next_s_goal_s[:, :self.env_size], dim=1)
        goal_state = F.softmax(next_s_goal_s[:, self.env_size:], dim=1)
        return pi, v, action, next_state, goal_state, h_out, c_out
