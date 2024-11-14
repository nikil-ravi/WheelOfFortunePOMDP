import random
from data.phrases import phrases
from data.wheel_values import wheel_values
from constants import *
from player import Player

import time

class WheelOfFortune:
    def __init__(self, phrases, wheel_values, num_players=2):
        self.phrases = phrases
        self.wheel_values = wheel_values
        self.players = [Player(i) for i in range(num_players)]
        self.reset_game() # to initialize the game

    def reset_game(self):
        self.current_phrase = random.choice(self.phrases).upper()
        self.current_player = 0 # TODO: should this be random?
        self.bankrupt = [False] * len(self.players)
        self.guessed_letters = set()

        # revealed phrase initially has underscores and spaces
        self.revealed_phrase = ['_' if letter.isalpha() else letter for letter in self.current_phrase]

        # tracks scores for each player- this is equivalent to accumulated "money" in the game
        self.scores = [0] * len(self.players)

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
            
        # Deduct the cost of the vowel from the current player's score
        self.scores[self.current_player] -= COST_OF_VOWEL
        
        # Make the guess and return the result
        return self.guess_letter(letter)

    def solve_puzzle(self, guess):
        guess = guess.upper()
        if guess == self.current_phrase:
            print("Congratulations! Player {} solved the puzzle!".format(self.current_player + 1))
            return True
        else:
            print("Incorrect solution.")
            return False

    def display_status(self):
        print("Phrase:", ' '.join(self.revealed_phrase))
        print(f"Scores: {', '.join([f'Player {i+1}: {score}' for i, score in enumerate(self.scores)])}")
        print(f"Guessed Letters: {', '.join(sorted(self.guessed_letters))}")

    def play_turn(self):
        current_player = self.players[self.current_player]
        print(f"-- Player {current_player.player_id + 1}'s Turn --")
        self.display_status()
        current_player.take_turn(self)

    def is_solved(self):
        return '_' not in self.revealed_phrase

    def start_game(self):
        print("Welcome to Wheel of Fortune!")
        while not self.is_solved():
            self.play_turn()
            if self.is_solved():
                break
            self.current_player = (self.current_player + 1) % len(self.players)
            #time.sleep(0.5)
        if self.is_solved():
            print("You have solved the puzzle!")
            self.display_status()
        else:
            print("Game over.")

if __name__ == "__main__":
    game = WheelOfFortune(phrases, wheel_values)
    game.start_game()
