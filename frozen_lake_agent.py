import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

class QLearningAgent:
    """Q-Learning agent for solving the Frozen Lake environment."""
    
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, 
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table; states x actions (filling with zeros)
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n)) # (rows: states, columns: actions)
        
    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        return np.argmax(self.q_table[state])  # Exploit (greedy)
    
    def update_q_table(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning formula."""
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Q-learning update
        self.q_table[state, action] = current_q + self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, episodes=10000):
        """Train the agent."""
        rewards_history = []
        success_history = []
        
        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.update_q_table(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            self.decay_epsilon()
            rewards_history.append(total_reward)
            success_history.append(1 if total_reward > 0 else 0)
            
            if (episode + 1) % 1000 == 0:
                avg_success = np.mean(success_history[-100:]) * 100
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Success Rate (last 100): {avg_success:.1f}% | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        return rewards_history, success_history
    
    def evaluate(self, episodes=100, render=False):
        """Evaluate the trained agent."""
        if render:
            eval_env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')
        else:
            eval_env = self.env
            
        successes = 0
        
        for episode in range(episodes):
            state, _ = eval_env.reset()
            done = False
            
            while not done:
                action = np.argmax(self.q_table[state])  # Pure exploitation
                state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                
                if done and reward > 0:
                    successes += 1
        
        if render:
            eval_env.close()
            
        return (successes / episodes) * 100
    
    def save(self, filepath='models/q_table.pkl'):
        """Save Q-table to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath='models/q_table.pkl'):
        """Load Q-table from file."""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Model loaded from {filepath}")


def plot_training_progress(rewards, success_history, window=100):
    """Plot training metrics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    
    # Plot success rate
    success_rate = [np.mean(success_history[max(0, i-window):i+1]) * 100 
                    for i in range(len(success_history))]
    ax1.plot(success_rate)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title(f'Success Rate (Moving Average, Window={window})')
    ax1.grid(True, alpha=0.3)
    
    # Plot rewards
    avg_rewards = [np.mean(rewards[max(0, i-window):i+1]) 
                   for i in range(len(rewards))]
    ax2.plot(avg_rewards)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.set_title(f'Average Reward (Moving Average, Window={window})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("Training plot saved to training_progress.png")
    plt.show()


if __name__ == "__main__":
    # Create environment
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    print("="*60)
    print("Frozen Lake Q-Learning Agent")
    print("="*60)
    print(f"State space: {env.observation_space.n}")
    print(f"Action space: {env.action_space.n}")
    print("="*60)
    
    # Create and train agent
    agent = QLearningAgent(
        env,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    print("\nTraining agent...")
    rewards, success_history = agent.train(episodes=10000)
    
    # Evaluate agent
    print("\n" + "="*60)
    print("Evaluating trained agent...")
    success_rate = agent.evaluate(episodes=100)
    print(f"Final Success Rate: {success_rate:.1f}%")
    print("="*60)
    
    # Save model and plot results
    agent.save('models/q_table.pkl')
    plot_training_progress(rewards, success_history)
    
    # Optionally render a few episodes
    print("\nRendering a few episodes (close window to continue)...")
    agent.evaluate(episodes=3, render=True)
    
    env.close()