import torch.nn as nn
import torch
import os
import random
import numpy as np
import argparse
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from evaluate import evaluate_HIV, evaluate_HIV_population
from torch.distributions import Categorical
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
    
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
    
        self.data[self.index] = (s, a, r, s_, d)
        self.index = int((self.index + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    
    def __len__(self):
        return len(self.data)
    
class Policy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, env, gamma=0.99):
        super(Policy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma


        self.fc1_val = nn.Linear(state_dim, hidden_dim)
        self.fc2_val = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_val = nn.Linear(hidden_dim, hidden_dim)
        self.fc4_val = nn.Linear(hidden_dim, hidden_dim)
        # self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6_val = nn.Linear(hidden_dim, action_dim)

        self.fc1_adv = nn.Linear(state_dim, hidden_dim)
        self.fc2_adv = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_adv = nn.Linear(hidden_dim, hidden_dim)
        self.fc4_adv = nn.Linear(hidden_dim, hidden_dim)
        # self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6_adv = nn.Linear(hidden_dim, action_dim)

        self.LeakyReLU = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.silu = nn.SiLU()

    def forward(self, x):
        val = self.silu(self.fc1_val(x))
        val = self.silu(self.fc2_val(val))
        val = self.silu(self.fc3_val(val))
        val = self.silu(self.fc4_val(val))
        # val = torch.leaky_relu(self.fc5(val))
        val = self.fc6_val(val)

        adv = self.silu(self.fc1_adv(x))
        adv = self.silu(self.fc2_adv(adv))
        adv = self.silu(self.fc3_adv(adv))
        adv = self.silu(self.fc4_adv(adv))
        # adv = torch.leaky_relu(self.fc5(adv))
        adv = self.fc6_adv(adv)


        # x = self.silu(self.fc1(x))
        # x = self.silu(self.fc2(x))
        # x = self.silu(self.fc3(x))
        # x = self.silu(self.fc4(x))
        # # x = torch.leaky_relu(self.fc5(x))
        # x = self.fc6(x)
        # x = self.tanh(x)
        # x = self.softmax(x)
        return val + adv - adv.mean()

    def sample_action(self, x):
        # probabilities = self.forward(x)
        # action_distribution = Categorical(probabilities)
        # return action_distribution.sample().item()
        return torch.argmax(self.forward(x).unsqueeze(0)).item()

    def log_prob(self, x, a):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.log_prob(a)

class ProjectAgent:
    def __init__(self):
        self.device = device

        self.batch_size = 1024
        self.gradient_steps = 2
        self.gamma = 0.98
        self.buffer_size = 1e6
        self.initial_buffer_size = 1024
        self.epsilon_min = 1e-2
        self.epsilon_max = 1.0
        self.epsilon = self.epsilon_max
        self.epsilon_stop = 1e4
        self.epsilon_delay = 4e2
        self.step = 0
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.episode = 0
        self.max_episode = 4000

        self.action_dim = 6
        self.hidden_dim = 512
        self.output_dim = 4
        self.learning_rate = 1e-3

        self.monitoring_nb_trials = 50
        self.tensorboard = ...

        self.update_strategy = 'ema'
        self.update_target_freq = 600
        self.update_target_tau = 1e-3

        self.save_frequency = 50

        self.model = Policy(self.action_dim, self.hidden_dim, self.output_dim, env).to(self.device)
        self.target_model = Policy(self.action_dim, self.hidden_dim, self.output_dim, env).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.default_save_path = "model.pth"


        self.memory = ReplayBuffer(self.buffer_size, device)

    def save(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'parameters': {
                'batch_size': self.batch_size,
                'gradient_steps': self.gradient_steps,
                'gamma': self.gamma,
                'buffer_size': self.buffer_size,
                'initial_buffer_size': self.initial_buffer_size,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_max': self.epsilon_max,
                'epsilon_stop': self.epsilon_stop,
                'epsilon_delay': self.epsilon_delay,
                'epsilon_step': self.epsilon_step,
                'step': self.step,
                'max_episode': self.max_episode,
                'episode': self.episode,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'learning_rate': self.learning_rate,
                'monitoring_nb_trials': self.monitoring_nb_trials,
                'update_strategy': self.update_strategy,
                'update_target_freq': self.update_target_freq,
                'update_target_tau': self.update_target_tau,
                'capacity': self.memory.capacity,
                'data': self.memory.data,
                'index': self.memory.index,
                'tensorboard_logdir': self.tensorboard.get_logdir(),
            },
        }
        torch.save(checkpoint, path)

    def load(self, path=None):
        if path==None:
            path = self.default_save_path
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            parameters = checkpoint['parameters']
            self.batch_size = parameters['batch_size']
            self.gradient_steps = parameters['gradient_steps']
            self.gamma = parameters['gamma']
            self.buffer_size = parameters['buffer_size']
            self.initial_buffer_size = parameters['initial_buffer_size']
            self.epsilon = parameters['epsilon']
            self.epsilon_min = parameters['epsilon_min']
            self.epsilon_max = parameters['epsilon_max']
            self.epsilon_stop = parameters['epsilon_stop']
            self.epsilon_delay = parameters['epsilon_delay']
            self.epsilon_step = parameters['epsilon_step']
            self.step = parameters['step']
            self.episode = parameters['episode']
            self.max_episode = parameters['max_episode']
            self.action_dim = parameters['action_dim']
            self.hidden_dim = parameters['hidden_dim']
            self.output_dim = parameters['output_dim']
            self.learning_rate = parameters['learning_rate']
            self.monitoring_nb_trials = parameters['monitoring_nb_trials']
            self.update_strategy = parameters['update_strategy']
            self.update_target_freq = parameters['update_target_freq']
            self.update_target_tau = parameters['update_target_tau']
            self.memory.capacity = parameters['capacity']
            self.memory.data = parameters['data']
            self.memory.index = parameters['index']
            # self.tensorboard = SummaryWriter(log_dir=parameters['tensorboard_logdir'])
            print(f"Agent loaded from episode {self.episode}.")
            return True
        else:
            print(f"No model found at {path}.")
            # self.tensorboard = SummaryWriter()
            return False

    def act(self, observation, use_random=False):
        # We use float32 for compatibility with MPS devices.
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

    # def save(self, path):
    #     torch.save(self.model.state_dict(), path)

    # def load(self):
    #     if os.path.isfile(self.default_save_path):
    #         self.model.load_state_dict(torch.load(self.default_save_path, map_location=self.device))
    #         print(f"Agent loaded on {device}.")
    #         return True
    #     else:
    #         print(f"No model found at {self.default_save_path}.")
    #         return False

    def V_initial_state(self, env, nb_trials): 
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)).max().item())
        return np.mean(val)
    
    def train(self, env):
        # Fill buffer
        if self.episode == 0:
            self.fill_memory(env)

        V_init_state = []
        episode_return = []
        episode_cum_reward = 0

        state,_ = env.reset()
        loss = ...
        while self.episode < self.max_episode:
            # Update epsilon 
            if self.step > self.epsilon_delay:
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

            # Select epsilon greedy action
            if random.random() > self.epsilon:
                action = self.act(state)
            else:
                action = env.action_space.sample()

            # Take action and observe reward and next state
            next_state, reward, done, truncated, _ = env.step(action)

            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # Gradient step
            if len(self.memory) >= self.batch_size:
                for _ in range(self.gradient_steps):
                    states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

                    evaluated_action = torch.argmax(self.model(next_states).detach(), dim=1)
                    QY = self.target_model(next_states).detach() 
                    QYmax = QY.gather(1, evaluated_action.unsqueeze(1)).squeeze(1)

                    update = torch.addcmul(rewards, 1 - dones, QYmax, value=self.gamma)
                    QXA = self.model(states).gather(1, actions.unsqueeze(1).long())

                    loss = self.criterion(QXA, update.unsqueeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            
            if self.update_strategy == 'replace':
                if self.step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            elif self.update_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                for k in target_state_dict.keys():
                    target_state_dict[k] = self.update_target_tau * model_state_dict[k] + (1 - self.update_target_tau) * target_state_dict[k]
                self.target_model.load_state_dict(target_state_dict)

            # Update state
            self.step += 1
            if done or truncated:
                self.episode += 1

                V0 = self.V_initial_state(env, self.monitoring_nb_trials)
                V_init_state.append(V0)

                # self.tensorboard.add_scalar("Reward/episode", episode_cum_reward, self.episode)
                # self.tensorboard.add_scalar("Epsilon/episode", self.epsilon, self.episode)
                # self.tensorboard.add_scalar("Loss/episode", loss, self.episode)
                # self.tensorboard.add_scalar("V0/episode", V0, self.episode)
                # self.tensorboard.flush()

                if self.episode % self.save_frequency == 0:
                    self.save(self.default_save_path)

                print(f"Episode {self.episode} - epsilon {self.epsilon:.5e} - batch size {len(self.memory)} - episode return {episode_cum_reward:.5e} - V0 {V0:.5e} - loss {loss:.5e}")
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        # self.tensorboard.close()     
        self.save(self.default_save_path)
        print("Training finished.")

        return episode_return, V_init_state
    
    def fill_memory(self, env):
        print(f"Filling memory with {self.initial_buffer_size} samples...")
        state,_ = env.reset()
        for _ in tqdm(range(int(self.initial_buffer_size))):
            action = env.action_space.sample()
            next_state, reward, done, truncated, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            if done or truncated:
                state, _ = env.reset()
            else:
                state = next_state

        print(f"Memory filled with {len(self.memory)} samples.")



def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", type=bool, default=False)
    args = parser.parse_args()
    force_train = args.f

    seed_everything(seed=42)


    agent = ProjectAgent()
    if not(agent.load()) or force_train:
        print("Training the agent...")
        score, v0 = agent.train(env)
        agent.save(agent.default_save_path)
        plt.plot(score, color="purple")
        # plt.plot(v0, color="magenta")
        plt.tight_layout()
        plt.savefig("score.pdf")
    

    print("Evaluating the agent...")
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
    with open(file="score.txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")
    print(f"Score of the agent: {score_agent:2e}")
    print(f"Score of the agent on the population: {score_agent_dr:.2e}")

