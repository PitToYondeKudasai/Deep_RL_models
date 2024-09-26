import numpy as np
import copy 
import matplotlib.pyplot as plt
import random

def get_states(n_row: int, n_col: int, size_px:int):
    px = (size_px+size_px//20)
    states_coord = px*np.array(np.meshgrid(np.arange(0.5, n_col), np.arange(0.5, n_row))).T.reshape(n_col*n_row, -1)
    states = {i : states_coord[i] for i in range(len(states_coord))}
    return states, px

def get_walls(states: dict, px: int) -> dict:
    DELTA = np.array([(px/2, px/2), (-px/2, px/2), (px/2, -px/2), (-px/2, -px/2), ])
    walls = {}
    for state in states:
        vertices = [states[state] + delta for delta in DELTA]
        walls[state] = {'UP': [vertices[0], vertices[1]], 
                        'RIGHT': [vertices[0], vertices[2]], 
                        'DOWN':[vertices[2], vertices[3]],
                        'LEFT': [vertices[1], vertices[3]]}
    return walls

def plot_walls(walls: dict, height: int, width: int, states: dict=None, labels: bool=False) -> None:
    fig, ax = plt.subplots(figsize=(width, height))
    for segments in walls.values():
        for direction in segments:
            x = np.array(segments[direction])[:, 0]
            y = np.array(segments[direction])[:, 1]
            ax.plot(x, y, color='red')
    if states is not None:
        for state in states.values():
            ax.scatter(state[0], state[1], color='blue')
    if labels:
        for label in states:
            ax.annotate(label, states[label]+3)
    return fig, ax   

def remove_wall(walls: dict, states: dict, connection: dict, px: int, height: int, width: int) -> dict:
    walls_copy = copy.deepcopy(walls)
    state1 = states[connection[0]]
    state2 = states[connection[1]]
    delta = state2 - state1
    if delta[0] == px:
        dir1 = 'RIGHT'
        dir2 = 'LEFT'
    elif delta[1] == px:
        dir1 = 'UP'
        dir2 = 'DOWN'
    elif delta[0] == -px:
        dir1 = 'LEFT'
        dir2 = 'RIGHT'
    elif delta[1] == -px:
        dir1 = 'DOWN'
        dir2 = 'UP'
    elif delta[0] == width-px:
        dir1 = 'LEFT'
        dir2 = 'RIGHT'
    elif delta[1] == height-px:
        dir1 = 'DOWN'
        dir2 = 'UP'
    elif delta[0] == -width+px:
        dir1 = 'RIGHT'
        dir2 = 'LEFT'
    elif delta[1] == -height+px:
        dir1 = 'UP'
        dir2 = 'DOWN'
    else:
        return walls_copy
    del walls_copy[connection[0]][dir1]
    del walls_copy[connection[1]][dir2]
    return walls_copy     

def upper_node(node_id, n_rows):
    return ((node_id - n_rows * (node_id // n_rows) - n_rows + 1) % (n_rows)) + n_rows*(node_id // n_rows)

def lower_node(node_id, n_rows):
    return ((node_id - n_rows * (node_id // n_rows) - n_rows - 1) % (n_rows)) + n_rows*(node_id // n_rows)

def right_node(node_id, n_cols, n_rows):
    next_col = ((n_cols + (node_id // n_rows) + 1) % n_cols)
    return next_col*n_rows + (node_id%n_rows)

def left_node(node_id, n_cols, n_rows):
    prev_col = ((n_cols + (node_id // n_rows) -1) % n_cols)
    return prev_col*n_rows + (node_id%n_rows)

def get_neighbors(n_cols: int, n_rows: int) -> dict:
    neighs = {}
    for node_id in range(n_cols * n_rows):
        neighs[node_id] = {'UP': upper_node(node_id, n_rows),
                           'RIGHT': right_node(node_id, n_cols, n_rows),
                           'DOWN': lower_node(node_id, n_rows), 
                           'LEFT': left_node(node_id, n_cols, n_rows)}
    return neighs

def generate_maze(n_cols: int, n_rows: int, env):
    s = np.random.choice(range(n_cols*n_rows))
    V = []
    A = []
    A, V = walk_maze(s, A, V, env)
    return A, V

def walk_maze(s, A, V, env):
    V.append(s)
    list_neighbors = list(env.neighbors[s].values())
    random.shuffle(list_neighbors)
    for n in list_neighbors:
        if n not in  V:
            A.append([s, n])
            A, V = walk_maze(n, A, V, env)
    return A, V

class Maze():
    def __init__(self, n_cols, n_rows, size_px=40, rew_goal=1, rew_step=0):
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.size_px = size_px  
        self.height = (self.size_px + self.size_px // 20) * (self.n_rows)
        self.width = (self.size_px + self.size_px // 20) * (self.n_cols)
        self.states, self.px = get_states(self.n_rows, self.n_cols, self.size_px)
        self.walls = get_walls(self.states, self.px)
        self.neighbors = get_neighbors(self.n_cols, self.n_rows)
        self.all_walls = []
        for node in self.neighbors:
            self.all_walls.extend([[node, self.neighbors[node][dir]] for dir in ['UP', 'RIGHT', 'DOWN', 'LEFT']])
        self.all_walls = np.array(self.all_walls)
        idx = np.unique(np.sort(self.all_walls), axis=0, return_index=True)[1]
        self.all_walls = self.all_walls[idx]

        self.agent_init_position = None

        self.actions = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3:'LEFT'}
        self.rew_goal = rew_goal
        self.rew_step = rew_step
        self.done = None

    def reset(self, reset_maze: bool=False, n_edges_remove: int=3, reset_agent_pos: bool=False, reset_goal_pos: bool=False):
        self.done = False
        if reset_maze or (self.agent_init_position is None):
            self.goal_position = np.random.choice(self.n_cols*self.n_rows)
            self.agent_init_position = np.random.choice(self.n_cols*self.n_rows)
            while self.agent_init_position == self.goal_position:
                self.agent_init_position = np.random.choice(self.n_cols*self.n_rows)
            self.agent_position = self.agent_init_position
            self.maze = copy.deepcopy(self.walls)
            self.walls_removed, _ = generate_maze(self.n_cols, self.n_rows, self)
            self.walls_removed = np.array(self.walls_removed)
            self.walls_removed = np.sort(self.walls_removed, 1)
            idx = np.unique(np.sort(self.walls_removed), axis=0, return_index=True)[1]
            self.walls_removed = self.walls_removed[idx]           
            for i in range(n_edges_remove):
                mask1 = np.sum(np.array([np.all(self.walls_removed[i] == self.all_walls, 1) 
                                        for i in range(self.walls_removed.shape[0])]), 0)
                mask2 = np.sum(np.array([np.all(self.walls_removed[:, [1,0]][i] == self.all_walls, 1) 
                                        for i in range(self.walls_removed.shape[0])]), 0)
                mask = mask1 + mask2
                new_walls_removed = self.all_walls[np.random.choice(np.where(mask==0)[0], 1)]
                self.walls_removed = np.concatenate([self.walls_removed, new_walls_removed ])
            for wall in self.walls_removed:
                self.maze = remove_wall(self.maze, self.states, wall, self.px, self.height, self.width)
            self.is_in_check = (self.walls_removed[:, None] == self.all_walls).all(axis=2).any(axis=0)
            self.adj_dict = {}
            for node in self.neighbors:
                self.adj_dict[node] = {}
                for direction in self.neighbors[node]:
                    node_neighbor = self.neighbors[node][direction]
                    if (np.sum(np.all(np.array([node, node_neighbor]) == self.walls_removed[np.argsort(self.walls_removed[:, 0])], 1)) +
                        np.sum(np.all(np.array([node_neighbor, node]) == self.walls_removed[np.argsort(self.walls_removed[:, 0])], 1))):
                        self.adj_dict[node][direction] = node_neighbor
                    else:
                        self.adj_dict[node][direction] = node
        else:
            if reset_agent_pos:
                self.agent_init_position = np.random.choice(self.n_cols*self.n_rows)
                while self.agent_init_position == self.goal_position:
                    self.agent_init_position = np.random.choice(self.n_cols*self.n_rows)            
            self.agent_position = self.agent_init_position
            if reset_goal_pos:
                self.goal_position = np.random.choice(self.n_cols*self.n_rows)
                while self.goal_position == self.agent_init_position:
                    self.goal_position = np.random.choice(self.n_cols*self.n_rows)   
        return self.agent_position, 0, self.done, {}

    def step(self, action):
        assert self.done == False
        rew = self.rew_step
        action = self.actions[action]
        self.agent_position = self.adj_dict[self.agent_position][action]
        if self.agent_position == self.goal_position:
            rew = self.rew_goal
            self.done = True
        return self.agent_position, rew, self.done, {}
        
    def plot_maze(self, height=5, width=5, states=True, with_agent=False, with_goal=False, sim_pos=None, save_path=None):
        if states:
            fig, ax = plot_walls(self.maze, height=height, width=width, states=self.states, labels=True)
        else:
            fig, ax = plot_walls(self.maze, height=height, width=width)
        if with_agent:
            agent_circle = plt.Circle(self.states[self.agent_position], 5, color='r', fill=True)
            ax.add_patch(agent_circle)
        if with_goal:
            goal_circle = plt.Circle(self.states[self.goal_position], 5, color='g', fill=False)
            ax.add_patch(goal_circle)
        if sim_pos is not None:
            sim_circle = plt.Circle(self.states[sim_pos], 5, color='r', fill=True, alpha=0.3)
            ax.add_patch(sim_circle)
        ax.axis('off')
        if  save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        else:
            return fig, ax
