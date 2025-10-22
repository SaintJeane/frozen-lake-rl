# Frozen Lake Q-Learning Agent 🧊

A simple reinforcement learning project implementing Q-Learning to solve OpenAI Gymnasium's Frozen Lake environment.

## 📖 Project Overview

This project demonstrates Q-Learning, a model-free reinforcement learning algorithm, applied to the Frozen Lake problem. The agent learns to navigate a frozen lake from start to goal while avoiding holes in the ice.

**Environment:** The agent must navigate a 4x4 grid where:
- **S** = Starting position
- **F** = Frozen surface (safe)
- **H** = Hole (game over)
- **G** = Goal (success!)

The ice is slippery, so the agent doesn't always move in the intended direction, making it a stochastic environment.

## 🎯 Key Concepts Demonstrated

- **Q-Learning Algorithm**: Temporal difference learning method
- **Epsilon-Greedy Exploration**: Balance between exploration and exploitation
- **Q-Table**: Stores learned state-action values
- **Hyperparameter Tuning**: Learning rate, discount factor, epsilon decay

## 🚀 Getting Started

### Prerequisites

```bash
pip install gymnasium numpy matplotlib
```
- Or if the repository is cloned, run the following command in the terminal:

```bash
pip install -r requirements.txt
```

- This will install the packages in the `requirements.txt` file.

### Running the Project

```bash
python frozen_lake_agent.py
```

This will:
1. Train the agent for 10,000 episodes
2. Display training progress every 1,000 episodes
3. Evaluate the trained agent
4. Save the Q-table to `models/q_table.pkl`
5. Generate a training progress plot
6. Render a few episodes visually

## 📊 Results

After training, you should see:
- Success rate progression during training
- Final evaluation metrics
- Training visualization saved as `training_progress.png`

Expected performance: **70-80% success rate** on the slippery version of Frozen Lake.

## 🔧 Customization

You can modify hyperparameters in the main script:

```python
agent = QLearningAgent(
    env,
    learning_rate=0.1,      # How much to update Q-values
    discount_factor=0.99,   # How much to value future rewards
    epsilon_start=1.0,      # Initial exploration rate
    epsilon_min=0.01,       # Minimum exploration rate
    epsilon_decay=0.995     # Exploration decay rate
)
```

## 📁 Project Structure

```
frozen-lake-rl/
├── frozen_lake_agent.py    # Main agent implementation
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── models/                 # Saved Q-tables
│   └── q_table.pkl
└── training_progress.png   # Training visualization
```

## 🧠 How Q-Learning Works

The Q-Learning algorithm updates the Q-table using the formula:

```
Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- **Q(s,a)** = Q-value for state-action pair
- **α** = Learning rate
- **r** = Reward received
- **γ** = Discount factor
- **s'** = Next state

## 📈 Next Steps

- Try different hyperparameters
- Implement Double Q-Learning
- Test on other Gymnasium environments
- Add Deep Q-Networks (DQN) for larger state spaces
- Compare with policy gradient methods

## 📚 References

- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)
- [Q-Learning Paper](https://link.springer.com/article/10.1007/BF00992698)

## 📝 License

MIT License - feel free to use this project for learning and experimentation!

---

⭐ If you found this helpful, consider giving it a star on GitHub!