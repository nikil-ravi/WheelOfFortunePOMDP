import random
import numpy as np

class QLearningPlayer(Player):
    def __init__(self, player_id, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(player_id)
        self.q_table = {}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def get_state(self, game):
        # Represent the game state as a tuple
        return (
            tuple(game.revealed_phrase),
            game.scores[self.player_id],
            tuple(sorted(game.guessed_letters))
        )

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(['S', 'B', 'P'])
        q_values = self.q_table.get(state, {})
        if not q_values:
            return random.choice(['S', 'B', 'P'])  # Default random action
        return max(q_values, key=q_values.get)

    def update_q_value(self, state, action, reward, next_state):
        # Update Q-value using the Q-learning formula
        current_q = self.q_table.get(state, {}).get(action, 0)
        next_max_q = max(self.q_table.get(next_state, {}).values(), default=0)
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q

    def take_turn(self, game):
        state = self.get_state(game)
        action = self.choose_action(state)
        print(f"Player {self.player_id + 1} chose action: {action}")

        if action == 'S':
            spin_result = game.spin_wheel()
            if spin_result in ["Bankrupt", "Lose a Turn"]:
                reward = -10
            else:
                guessed_letter = self.choose_consonant()
                success = game.guess_letter(guessed_letter)
                reward = spin_result if success else -5
        elif action == 'B':
            letter = self.choose_vowel()
            success = game.buy_vowel(letter)
            reward = 5 if success else -5
        elif action == 'P':
            guess = self.choose_solution()
            success = game.solve_puzzle(guess)
            reward = 50 if success else -10
        else:
            print("Invalid action.")
            reward = -1

        next_state = self.get_state(game)
        self.update_q_value(state, action, reward, next_state)
