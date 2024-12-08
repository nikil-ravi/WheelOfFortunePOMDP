import random
from collections import defaultdict
import numpy as np
from players.player import Player
from data.phrases import phrases

def default_q_values():
    return np.zeros(3)

class QLearningPlayer(Player):
    def __init__(self, player_id, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        super().__init__(player_id)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(default_q_values)
        self.all_phrases = phrases

    def encode_state(self, game):
        revealed_phrase = ''.join(game.revealed_phrase)
        guessed_letters = ''.join(sorted(game.guessed_letters))
        score = game.scores[self.player_id]
        return (revealed_phrase, guessed_letters, score)
    def act(self, game):
        state = self.encode_state(game)
        available_actions = self.get_available_actions(game)
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            # Initialize Q-values for new states
            if state not in self.q_table:
                self.q_table[state] = np.zeros(3)
            return max(available_actions, key=lambda a: self.q_table[state][a])
            
    def only_vowels_available(self, game):
        possible_phrases = self.get_possible_phrases(game)
        # print(f"Possible phrases: {possible_phrases}")
        if not possible_phrases:
            print("No possible phrases found.")
            return False
        # print("Possible phrases found: ", possible_phrases)
        print("Computing letter counts...")
        letter_counts = defaultdict(int)
        print(f"Guessed letters: {game.guessed_letters}")
        for phrase in possible_phrases:
            for letter in set(phrase.upper()) - set(game.guessed_letters) - set('AEIOU'):
                letter_counts[letter] += 1
        print(f"Letter counts: {letter_counts}")
        return False if letter_counts else True

    def get_available_actions(self, game):
        actions = []
        if set('BCDFGHJKLMNPQRSTVWXYZ') - set(game.guessed_letters):
            actions.append(0)  # Spin (guess consonant)
        if set('AEIOU') - set(game.guessed_letters) and game.scores[self.player_id] >= 250:
            actions.append(1)  # Buy vowel
        actions.append(2)  # Solve puzzle is always available

        return actions if actions else [2]  # If no other options, must solve

    def update_q_table(self, state, action, reward, next_state, done):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)
        
        current_q = self.q_table[state][action]
        
        if done:
            max_next_q = 0
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(3)
            max_next_q = max(self.q_table[next_state])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def get_possible_phrases(self, game):
        revealed_phrase = ''.join(game.revealed_phrase)
        possible_solutions = []

        for phrase in game.phrases:
            if phrase.upper() in game.guessed_phrases:
                continue

            if len(phrase) != len(revealed_phrase):
                continue

            matches_pattern = True
            for revealed_char, phrase_char in zip(revealed_phrase, phrase):
                if revealed_char != '_':
                    if revealed_char.upper() != phrase_char.upper():
                        matches_pattern = False
                        break
                elif phrase_char.upper() in revealed_phrase.upper():
                    # If the letter is revealed elsewhere, it should be revealed here too
                    matches_pattern = False
                    break

            if not matches_pattern:
                continue

            consistent_with_guesses = True
            for letter in game.guessed_letters:
                if (letter.upper() in phrase.upper()) != (letter.upper() in revealed_phrase.upper()):
                    consistent_with_guesses = False
                    break
            if not consistent_with_guesses:
                continue

            possible_solutions.append(phrase)

        return possible_solutions

    def choose_consonant(self, game):
        possible_phrases = self.get_possible_phrases(game)
        # print(f"Possible phrases: {possible_phrases}")
        if not possible_phrases:
            print("No possible phrases found.")
            return None
        # print("Possible phrases found: ", possible_phrases)
        # print("Computing letter counts...")
        letter_counts = defaultdict(int)
        # print(f"Guessed letters: {game.guessed_letters}")
        for phrase in possible_phrases:
            for letter in set(phrase.upper()) - set(game.guessed_letters) - set('AEIOU'):
                letter_counts[letter] += 1
        # print(f"Letter counts: {letter_counts}")
        return max(letter_counts, key=letter_counts.get) if letter_counts else None

    def choose_vowel(self, game):
        possible_phrases = self.get_possible_phrases(game)
        if not possible_phrases:
            return None
        vowel_counts = defaultdict(int)
        unguessed_vowels = set('AEIOU') - set(game.guessed_letters)
        for phrase in possible_phrases:
            for letter in set(phrase.upper()) & unguessed_vowels:
                vowel_counts[letter] += 1
        return max(vowel_counts, key=vowel_counts.get) if vowel_counts else None

    def choose_solution(self, game):
        possible_phrases = self.get_possible_phrases(game)
        return random.choice(possible_phrases) if possible_phrases else game.current_phrase

    def spin(self, game):
        spin_result = game.spin_wheel()
        print(f"Player {self.player_id + 1} spun the wheel and got: {spin_result}")
        
        available_consonants = set('BCDFGHJKLMNPQRSTVWXYZ') - set(game.guessed_letters)
        # print(f"Available consonants: {', '.join(sorted(available_consonants))}")
        
        if spin_result == "Bankrupt":
            game.scores[self.player_id] = 0
            return -200  # High penalty for bankruptcy
        elif spin_result == "Lose a Turn":
            return -50  # Penalty for losing a turn
        else:
            letter = self.choose_consonant(game)
            if letter is None:
                print(f"Player {self.player_id + 1} couldn't choose a consonant.")
                return -20  # Penalty for not being able to choose a consonant
            print(f"Player {self.player_id + 1} guesses consonant: {letter}")
            if game.guess_letter(letter):
                occurrences = game.current_phrase.count(letter)
                points = spin_result * occurrences
                game.scores[self.player_id] += points
                bonus = 100 * occurrences  # High bonus for correct guess, scaled by occurrences
                return points + bonus  # Return both the points from the wheel and the bonus
            else:
                return -30  # Penalty for incorrect guess
            
    def buy_vowel(self, game):
        if game.scores[self.player_id] >= 250:  # Assuming 250 is the cost of a vowel
            letter = self.choose_vowel(game)
            if letter is None:
                print(f"Player {self.player_id + 1} couldn't choose a vowel.")
                return -20  # Penalty for not being able to choose a vowel
            print(f"Player {self.player_id + 1} buys vowel: {letter}")
            if game.buy_vowel(letter):
                occurrences = game.current_phrase.count(letter)
                bonus = 200 * occurrences  # High bonus for correct guess, scaled by occurrences
                return bonus - 250  # Return the bonus minus the cost of buying a vowel
            else:
                return -100  # Higher penalty for incorrect vowel guess
        else:
            print(f"Player {self.player_id + 1} doesn't have enough points to buy a vowel.")
            return -40  # Penalty for attempting to buy without enough points

    def solve_puzzle(self, game):
        possible_phrases = self.get_possible_phrases(game)
        unguessed_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') - set(game.guessed_letters)
        no_letters_available = not (set('BCDFGHJKLMNPQRSTVWXYZ') & unguessed_letters) and not (set('AEIOU') & unguessed_letters)

        guess = random.choice(possible_phrases) if possible_phrases else game.current_phrase
        print(f"Player {self.player_id + 1} attempts to solve: {guess}")
        if game.solve_puzzle(guess):
            return 2000  # High reward for solving the puzzle
        else:
            if no_letters_available:
                return -50  # Smaller penalty if no letters are available
            else:
                return -500  # Large penalty for incorrect solution attempt when letters are available

    def take_turn(self, game):
        state = self.encode_state(game)
        action = self.act(game)
        print(f"\nPlayer {self.player_id + 1}'s turn:")
        if action == 0:
            print("Action: Spin the wheel and guess a consonant")
            reward = self.spin(game)
        elif action == 1:
            print("Action: Buy a vowel")
            reward = self.buy_vowel(game)
        else:
            print("Action: Solve the puzzle")
            reward = self.solve_puzzle(game)
        next_state = self.encode_state(game)
        done = game.is_solved()
        self.update_q_table(state, action, reward, next_state, done)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        print(f"Reward for this action: {reward}")
        print(f"Current game state: {''.join(game.revealed_phrase)}")
        print(f"Player {self.player_id + 1}'s current score: {game.scores[self.player_id]}")

    def end_game(self, game):
        final_state = self.encode_state(game)
        final_reward = game.scores[self.player_id]
        for action in range(3):
            self.update_q_table(final_state, action, final_reward, final_state, True)