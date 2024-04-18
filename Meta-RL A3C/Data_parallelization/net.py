import torch 
import torch.nn as nn
import torch.nn.functional as F

class ACModel(nn.Module):
    def __init__(self, input_dim, action_dim, mem_out_size=128, activation=nn.ReLU()):
        super(ACModel, self).__init__()  

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            activation
        ) 
        
        self.memory = nn.LSTM(64, mem_out_size)

        self.fc_pi = nn.Sequential(
            nn.Linear(mem_out_size, 32),
            activation,
            nn.Linear(32, action_dim)
        )
        
        self.fc_v = nn.Sequential(
            nn.Linear(mem_out_size, 32),
            activation,
            nn.Linear(32, 1)
        )
               
    def forward(self, *args):
        x = args[0]
        h_in, c_in = args[-2:]
        
        x = self.fc(x)
        out, (h_out, c_out) = self.memory(x, (h_in, c_in))
        
        v = self.fc_v(out)
        pi = self.fc_pi(out)
        pi = F.softmax(pi, dim=2).squeeze(1)
        
        return pi, v, h_out, c_out