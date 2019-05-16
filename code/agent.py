from .config import Config
from collections import deque, namedtuple
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

class AgentEval:
    """
        Agent that implement ONLY the evaluation mode, is usefull in a production environment.
    """
    def __init__(self, network, state_size, action_size, *vargs, **kwargs):
        """
            Contructor of environment
            Params:
                - network: architecture used by the agent.
                - state_size: number of the width state space.
                - action_size: number of the action space.
        """
        self.state_size = state_size
        self.action_size = action_size

        """
            NETWORK DEFINITION
        """
        self.net = network(self.state_size, self.action_size, *vargs, **kwargs).to(Config.DEVICE)

    def act(self, state):
        """
            Select the best action using Policy (that maximize reward from q(s, a))
            Params:
                - state: current state, a numpy array of (1, 37)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(Config.DEVICE)
        self.net.eval()
        with torch.no_grad():
            action_values = self.net(state)
        self.net.train()

        return np.argmax(action_values.cpu().data.numpy())

    def load(self, weights_path):
        """
            Allow to load weights from a file
            Params:
                - weights_path: path of the weights.
        """
        self.net.load_state_dict(torch.load(weights_path))
        self.net.eval()
        

class Agent(AgentEval):
    """
        Full implementation of the Agent, that includes the train process.
    """
    class PriorizedExpRepl:
        """
            Implementation of the Buffer Priorized Experience Replay
        """

        """
            Fast implementation of experience
        """
        Experience = namedtuple("Experience", field_names=[
            "error",
            "state", 
            "action", 
            "reward", 
            "next_state", 
            "done"
        ])

        def __init__(self, a, b, eps, buffer_size, batch_size):
            """
                Constructor of Buffer Priorized Experience Replay
                Params:
                    - a: hyperparameter that regularize the probability 
                         distribution based on absolute error (0-> uniform, 1->only depends of abs. error). 
                    - b: hyperparameter that regularize the importante of each experience during learning.
                    - eps: laplacian smooth to force to include in probability distribution cases that have 0 error.
                    - buffer_size: number of experiencies that buffer contain.
                    - batch_size: number of experiences that generate at the same time.
            """
            self.a = a
            self.b = b
            self.eps = eps
            self.memory = deque(maxlen=buffer_size)
            self.batch_size = batch_size

        def __get_samples(self):
            """
                Get samples from buffer
            """
            def samples_to_torch(experiences):
                """
                    Convert experience to torch tensor
                """
                errors, states, actions, rewards, next_states, dones = list(zip(*filter(lambda x: x is not None, experiences)))
                errors = torch.from_numpy(np.stack(errors)).float().to(Config.DEVICE)
                states = torch.from_numpy(np.stack(states)).float().to(Config.DEVICE)
                actions = torch.from_numpy(np.stack(actions)).long().to(Config.DEVICE)
                rewards = torch.from_numpy(np.stack(rewards)).float().to(Config.DEVICE)
                next_states = torch.from_numpy(np.stack(next_states)).float().to(Config.DEVICE)
                dones = torch.from_numpy(np.stack(dones).astype(np.uint8)).float().to(Config.DEVICE)

                return errors, states, actions, rewards, next_states, dones

            # Convert abs error to a probability distribution
            probs = np.array([(exp.error + self.eps) ** self.a for exp in self.memory])
            
            # Gibbs measure
            probs = probs / np.sum(probs)
            
            # Random choice using probabilities
            batch_idx = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, p=probs)

            # Generate the asociated weights of each selected experience
            weights_coef = (1. / (self.batch_size * probs[batch_idx])) ** self.b

            return batch_idx, torch.tensor(weights_coef).to(Config.DEVICE), samples_to_torch(map(self.memory.__getitem__, batch_idx))
 
        def samples(self):
            """
                Return samples if the memory are enough full
            """
            if len(self.memory) >= self.batch_size:
                return self.__get_samples()
            else:
                return None, None, None

        def add(self, error, state, action, reward, next_state, done):
            """
                Add new experience inside the buffer
                Params:
                    - error: error of the experience using a evaluated version of network.
                    - state: state of the experience
                    - action: action of the experience.
                    - reward: reward of the experience.
                    - next_state: next_state of the experience.
                    - done: if the episode is done of the experience.
            """
            exp = self.Experience(error, state, action, reward, next_state, done)
            self.memory.append(exp)

        def update_error(self, idx, error):
            """
                Update the absolute error of the specified experience
                Params:
                    - idx: index of the experience.
                    - error: new error of the experience using a evaluated version of network.
            """
            self.memory[idx] = self.memory[idx]._replace(error=error)

    def __init__(self, network, state_size, action_size, buffer_size=Config.BUFFER_SIZE, *vargs, **kwargs):
        super().__init__(network, state_size, action_size, *vargs, **kwargs)
        """
            NETWORK DEFINITION OF "COPY" NETWORK
        """
        self.net_fixed = network(self.state_size, self.action_size, *vargs, **kwargs).to(Config.DEVICE)

        """
            TRAINING DEFINITION
        """
        self.optimizer = optim.Adam(self.net.parameters(), lr=Config.LR)
        self.iteration = 0

        """
            BUFFER REPLAY IMPLEMENTATION
        """
        self.buffer = Agent.PriorizedExpRepl(Config.BUFFER_A, 
            Config.BUFFER_B, 
            Config.BUFFER_EPS, 
            buffer_size, 
            Config.BATCH_SIZE)

    def __get_error(self, state, action, reward, next_state, done, gamma):
        """
            Compute the actual error using both network in a evaluate mode.
            Params:
                - state: state of the experience
                - action: action of the experience.
                - reward: reward of the experience.
                - next_state: next_state of the experience.
                - done: if the episode is done of the experience.
                - gamma: discount factor.
        """
        states = torch.from_numpy(state).float().unsqueeze(0).to(Config.DEVICE)
        actions = torch.from_numpy(action).long().unsqueeze(0).to(Config.DEVICE)
        rewards = torch.from_numpy(reward).float().unsqueeze(0).to(Config.DEVICE)
        next_states = torch.from_numpy(next_state).float().unsqueeze(0).to(Config.DEVICE)
        dones = torch.from_numpy(done.astype(np.uint8)).float().unsqueeze(0).to(Config.DEVICE)

        self.net.eval()
        with torch.no_grad():
            # Double DQN evaluation error only
            _, best_actions = self.net(next_states).max(dim=-1)
            q_next = rewards + gamma * self.net_fixed(next_states).gather(-1, best_actions.unsqueeze(-1)) * (1 - dones)
            q = self.net(states).gather(-1, actions)

            errors = (q_next.detach() - q).abs().sum()
        self.net.train()

        return float(errors)

    def step(self, state, action, reward, next_state, done):
        """
            Compute the current step of the agent.
            Params:
                - state: current state.
                - action: current action.
                - reward: asociated reward.
                - next_state: next_state.
                - done: if the episode is done.
        """
        error = self.__get_error(state, action, reward, next_state, done, Config.GAMMA)
        self.buffer.add(error, state, action, reward, next_state, done)
        self.iteration += 1

        # Only every X step the network learns.
        if self.iteration % Config.UPDATE_EVERY == 0:
            indices, weights_coef, experiences = self.buffer.samples()
            if experiences is not None:
                self.learn(indices, weights_coef, experiences, Config.GAMMA)
 
    def learn(self, indices, weights_coef, experiences, gamma):
        """
            Learn the agent.
            Params:
                - indices: indices of the experiences.
                - action: current action.
                - weights_coef: coeficients of each experience.
                - experiences: batch of experiences.
                - gamma: discount factor.
        """
        _, states, actions, rewards, next_states, dones = experiences

        # Double DQN
        _, best_actions = self.net(next_states).max(dim=-1)
        q_next = rewards + gamma * self.net_fixed(next_states).gather(-1, best_actions.unsqueeze(-1)) * (1 - dones)

        q = self.net(states).gather(-1, actions)

        # Obtain priorized buffer error
        errors = (q_next.detach() - q)
        loss = (weights_coef.type(errors.dtype) * (errors ** 2).sum(dim=-1)).sum()

        # Update experiences to buffer
        abs_error = errors.abs().sum(dim=-1).detach().cpu().numpy()
        for i in range(Config.BATCH_SIZE):
            self.buffer.update_error(indices[i], abs_error[i])

        # Make backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update net_fixed
        self.soft_update(self.net, self.net_fixed, Config.TAU)

    def act(self, state, eps=0.):
        """
            Select an action in learning process. Allow to use an Epsilon-greedy action to force exploration env.
            Params:
                - indices: indices of the experiences.
                - action: current action.
                - weights_coef: coeficients of each experience.
                - experiences: batch of experiences.
                - gamma: discount factor.
        """
        if random.random() > eps:
            return super().act(state)
        else:
            return random.choice(np.arange(self.action_size))

    def soft_update(self, net, net_fixed, tau):
        """
            Update the net_fixed weights using the weight of net.
            Params:
                - net: current network
                - net_fixed: network with fixed weights.
                - tau: online avg balance parameter.
        """
        for param, param_fixed in zip(net.parameters(), net_fixed.parameters()):
            param_fixed.data.copy_(tau * param.data + (1.0 - tau) * param_fixed.data)

    def save(self, weights_path):
        """
            Save the model
            Params:
                - weights_path: path of the weights to save.
        """
        torch.save(self.net.state_dict(), weights_path)

    def load(self, weights_path):
        """
            Allow to load weights from a file
            Params:
                - weights_path: path of the weights.
        """
        self.net.load_state_dict(torch.load(weights_path))
        self.net_fixed.load_state_dict(torch.load(weights_path))
    

