import random
import pickle
from data.phrases import phrases
from data.words import words
from data.wheel_values import wheel_values
from constants import *
from players.player import Player
from players.naive_player import NaivePlayer
from players.q_learning_player import QLearningPlayer

import time

class WheelOfFortune:
    def __init__(self, phrases, wheel_values, num_players=2, player_class=Player):
        self.phrases = phrases
        self.wheel_values = wheel_values
        self.players = [player_class(i) for i in range(num_players)]
        
        # Do not reset game here. We'll do it in start_game().
        self.current_phrase = None
        self.current_player = 0
        self.bankrupt = [False] * len(self.players)
        self.guessed_letters = set()
        self.guessed_phrases = set()
        self.revealed_phrase = []
        self.scores = [0] * len(self.players)
        self.turn_count = 0

    def reset_game(self):
        self.current_phrase = random.choice(self.phrases).upper()
        print(f"The word is: {self.current_phrase}. Let's play!")
        self.current_player = 0
        self.bankrupt = [False] * len(self.players)
        self.guessed_letters = set()
        self.guessed_phrases = set()

        self.revealed_phrase = ['_' if letter.isalpha() else letter for letter in self.current_phrase]
        self.scores = [0] * len(self.players)
        self.turn_count = 0

    def spin_wheel(self):
        outcome = random.choice(self.wheel_values)
        print(f"Wheel spin outcome: {outcome}")
        return outcome

    def guess_letter(self, letter):
        letter = letter.upper()
        if letter in self.guessed_letters:
            print(f"{letter} has already been guessed.")
            return False

        self.guessed_letters.add(letter)
        if letter in self.current_phrase:
            occurrences = self.current_phrase.count(letter)
            print(f"{letter} is in the phrase! It occurs {occurrences} time(s).")
            self.revealed_phrase = [
                letter if self.current_phrase[i] == letter else self.revealed_phrase[i]
                for i in range(len(self.current_phrase))
            ]
            return True
        else:
            print(f"{letter} is not in the phrase.")
            return False

    def buy_vowel(self, letter):
        if self.scores[self.current_player] < COST_OF_VOWEL:
            print("Not enough score to buy a vowel.")
            return False

        letter = letter.upper()
        if letter not in VOWELS:
            print("You can only buy vowels.")
            return False

        self.scores[self.current_player] -= COST_OF_VOWEL
        return self.guess_letter(letter)

    def solve_puzzle(self, guess):
        guess = guess.upper()

        if guess == self.current_phrase:
            print("Congratulations! Player {} solved the puzzle!".format(self.current_player + 1))
            self.revealed_phrase = list(self.current_phrase)
            print("The phrase was:", ' '.join(self.revealed_phrase))
            return True
        else:
            print("Incorrect solution. Player guessed: {}".format(guess))
            self.guessed_phrases.add(guess)
            return False

    def display_status(self):
        print("Phrase:", ' '.join(self.revealed_phrase))
        print(f"Scores: {', '.join([f'Player {i+1}: {score}' for i, score in enumerate(self.scores)])}")
        print(f"Guessed Letters: {', '.join(sorted(self.guessed_letters))}")
        print(f"Guessed Phrases: {', '.join(sorted(self.guessed_phrases))}")

    def play_turn(self):
        self.turn_count += 1
        current_player = self.players[self.current_player]
        print(f"-- Player {current_player.player_id + 1}'s Turn (Turn count: {self.turn_count}) --")
        self.display_status()
        current_player.take_turn(self)

    def is_solved(self):
        return '_' not in self.revealed_phrase

    def start_game(self):
        # Now we call reset_game() here, ensuring only one phrase selection per game
        self.reset_game()
        print("Welcome to Wheel of Fortune!")
        while not self.is_solved():
            self.play_turn()
            if self.is_solved():
                break
            self.current_player = (self.current_player + 1) % len(self.players)
            #time.sleep(0.5)
        if self.is_solved():
            print("Congrats to Player {} for solving the puzzle!".format(self.current_player + 1))
        else:
            print("Game over.")
        print(f"Total turns taken: {self.turn_count}")

if __name__ == "__main__":
    import time

    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)

    def q_player_factory(pid):
        player = QLearningPlayer(pid, alpha=0.1, gamma=0.9, epsilon=0.0)  # zero epsilon for exploitation
        player.q_table = q_table
        return player

    total_turns = 0
    total_score = 0
    num_games = 1000

    start_time = time.time()
    for i in range(num_games):
        game = WheelOfFortune(words, wheel_values, num_players=1, player_class=q_player_factory)
        game.start_game()
        total_turns += game.turn_count
        total_score += game.scores[0]
    end_time = time.time()

    average_turns = total_turns / num_games
    average_score = total_score / num_games
    total_time = end_time - start_time
    average_time_per_game = total_time / num_games

    print(f"Average number of turns over {num_games} games: {average_turns:.2f}")
    print(f"Average final score over {num_games} games: {average_score:.2f}")
    print(f"Average time per game: {average_time_per_game:.4f} seconds")

