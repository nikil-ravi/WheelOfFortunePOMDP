import random
import numpy as np
from players.player import Player
from constants import VOWELS, CONSONANTS, COST_OF_VOWEL

class QLearningPlayer(Player):
    def __init__(self, player_id, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(player_id)
        self.q_table = {}
        self.alpha = alpha   # Learning rate
        self.gamma = gamma   # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def get_state(self, game):
        # Represent the game state as a tuple
        return (
            tuple(game.revealed_phrase),
            game.scores[self.player_id],
            tuple(sorted(game.guessed_letters))
        )

    def get_legal_actions(self, game):
        # Start with spin always allowed
        legal_actions = ['S']

        # Check if buying a vowel is possible
        available_vowels = [v for v in VOWELS if v not in game.guessed_letters]
        if game.scores[self.player_id] >= COST_OF_VOWEL and available_vowels:
            legal_actions.append('B')

        # Determine how many letters are revealed
        revealed_letters = sum(1 for c in game.revealed_phrase if c != '_')
        total_letters = sum(1 for c in game.current_phrase if c.isalpha())

        # Check if any consonants remain
        available_consonants = [c for c in CONSONANTS if c not in game.guessed_letters]

        # Allow solve under these conditions:
        # 1. If a significant portion (70%) of letters are revealed, OR
        # 2. If no vowels can be bought and no consonants remain, solving might be the only option.
        # This ensures the player won't get stuck at the end.
        if total_letters > 0:
            if revealed_letters > total_letters * 0.7:
                legal_actions.append('P')
            elif not available_vowels and not available_consonants:
                # No more letters to guess through spinning or buying vowels, must try solving
                legal_actions.append('P')

        return legal_actions

    def choose_action(self, state, game):
        legal_actions = self.get_legal_actions(game)

        # If no legal actions other than spin, it must spin. 
        # (Shouldn't happen, since spin is always allowed, but just in case)
        if not legal_actions:
            return 'S'

        # Epsilon-greedy action selection among legal actions
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        q_values = self.q_table.get(state, {})
        if not q_values:
            # If no Q-values exist for this state, choose randomly among legal actions
            return random.choice(legal_actions)

        # Filter Q-values to only include legal actions
        filtered_q_values = {a: q_values.get(a, 0) for a in legal_actions}
        # Pick the action with the highest Q-value among the legal ones
        return max(filtered_q_values, key=filtered_q_values.get)

    def update_q_value(self, state, action, reward, next_state):
        # Update Q-value using Q-learning formula
        current_q = self.q_table.get(state, {}).get(action, 0)
        next_max_q = max(self.q_table.get(next_state, {}).values(), default=0)
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q

    def choose_consonant(self):
        available_consonants = [c for c in CONSONANTS if c not in self.game.guessed_letters]
        return random.choice(available_consonants) if available_consonants else None

    def choose_vowel(self):
        available_vowels = [v for v in VOWELS if v not in self.game.guessed_letters]
        return random.choice(available_vowels) if available_vowels else None

    def choose_solution(self):
        # Currently random from possible solutions - could refine further
        possible_solutions = self.get_possible_solutions()
        return random.choice(possible_solutions) if possible_solutions else ""

    def get_possible_solutions(self):
        revealed_phrase = ''.join(self.game.revealed_phrase)
        possible_solutions = []

        for phrase in self.game.phrases:
            if phrase.upper() in self.game.guessed_phrases:
                continue
            if len(phrase) != len(revealed_phrase):
                continue
            matches_pattern = True
            for c1, c2 in zip(revealed_phrase, phrase):
                if c1 != '_' and c1.upper() != c2.upper():
                    matches_pattern = False
                    break
            if not matches_pattern:
                continue

            # Check consistency with guessed letters
            consistent_with_guesses = True
            for letter in self.game.guessed_letters:
                phrase_has_letter = letter.upper() in phrase.upper()
                revealed_has_letter = letter.upper() in revealed_phrase.upper()
                if phrase_has_letter != revealed_has_letter:
                    consistent_with_guesses = False
                    break
            if not consistent_with_guesses:
                continue

            possible_solutions.append(phrase)

        return possible_solutions

    def take_turn(self, game):
        self.game = game  # Update reference to current game state
        state = self.get_state(game)
        action = self.choose_action(state, game)
        print(f"Player {self.player_id + 1} chose action: {action}")

        if action == 'S':
            spin_result = game.spin_wheel()
            if spin_result in ["Bankrupt", "Lose a Turn"]:
                reward = -10
            else:
                guessed_letter = self.choose_consonant()
                if guessed_letter is None:
                    # No consonants available
                    reward = -5
                else:
                    success = game.guess_letter(guessed_letter)
                    # If correct, reward = spin_result + small bonus, else -5
                    reward = (spin_result + 5) if success else -5

        elif action == 'B':
            letter = self.choose_vowel()
            if letter is None:
                # No vowels available
                reward = -5
            else:
                success = game.buy_vowel(letter)
                # Small positive reward if successful, else -5
                reward = 5 if success else -5

        elif action == 'P':
            guess = self.choose_solution()
            success = game.solve_puzzle(guess)
            # Larger penalty for incorrect guess to discourage random solves
            reward = 100 if success else -100
        else:
            # Unrecognized action
            reward = -1

        next_state = self.get_state(game)
        self.update_q_value(state, action, reward, next_state)
