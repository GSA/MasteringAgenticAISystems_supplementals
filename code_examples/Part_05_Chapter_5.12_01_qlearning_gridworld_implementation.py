import numpy as np

class GridWorld:
    """4x4 grid world environment with goal and trap"""

    def __init__(self):
        self.grid_size = 4
        self.start = (0, 0)  # Bottom-left corner
        self.goal = (3, 3)   # Top-right corner
        self.trap = (2, 2)   # Center trap
        self.state = self.start

    def reset(self):
        """Start new episode from initial state"""
        self.state = self.start
        return self.state

    def step(self, action):
        """Execute action, return (new_state, reward, done)"""
        # Actions: 0=up, 1=down, 2=left, 3=right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_row = self.state[0] + moves[action][0]
        new_col = self.state[1] + moves[action][1]

        # Check boundaries - stay in place if hitting wall
        if new_row < 0 or new_row >= self.grid_size or \
           new_col < 0 or new_col >= self.grid_size:
            return self.state, -1, False  # Small penalty, episode continues

        # Update state
        self.state = (new_row, new_col)

        # Determine reward
        if self.state == self.goal:
            return self.state, 100, True   # Large reward, episode ends
        elif self.state == self.trap:
            return self.state, -100, True  # Large penalty, episode ends
        else:
            return self.state, -1, False   # Small step penalty

class QLearningAgent:
    """Agent learning optimal policy through Q-learning"""

    def __init__(self, grid_size, n_actions):
        # Initialize Q-table: Q[state][action] = estimated value
        self.Q = np.zeros((grid_size, grid_size, n_actions))
        self.alpha = 0.1    # Learning rate
        self.gamma = 0.9    # Discount factor
        self.epsilon = 1.0  # Exploration rate (starts high)

    def select_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.randint(4)
        else:
            # Exploit: choose best known action
            return np.argmax(self.Q[state[0], state[1]])

    def update(self, state, action, reward, next_state, done):
        """Update Q-value based on experience"""
        current_q = self.Q[state[0], state[1], action]

        if done:
            # Terminal state: no future value
            target = reward
        else:
            # Best possible future value from next state
            max_next_q = np.max(self.Q[next_state[0], next_state[1]])
            target = reward + self.gamma * max_next_q

        # Update Q-value toward target
        self.Q[state[0], state[1], action] += \
            self.alpha * (target - current_q)

# Training loop
env = GridWorld()
agent = QLearningAgent(grid_size=4, n_actions=4)
episodes = 1000

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(100):  # Max 100 steps per episode
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state

        if done:
            break

    # Decay exploration rate over time
    agent.epsilon = max(0.01, agent.epsilon * 0.995)

    if episode % 100 == 0:
        print(f"Episode {episode}: Total Reward = {total_reward}")
