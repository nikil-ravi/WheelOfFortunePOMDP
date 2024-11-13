import random
from data.phrases import phrases
from data.wheel_values import wheel_values
from constants import *

class WheelOfFortune:
    def __init__(self, phrases, wheel_values, num_players=2):
        self.phrases = phrases
        self.wheel_values = wheel_values
        self.num_players = num_players
        self.reset_game() # to initialize the game

    def reset_game(self):
        self.current_phrase = random.choice(self.phrases).upper()
        self.current_player = 0 # TODO: should this be random?
        self.bankrupt = [False] * self.num_players
        self.guessed_letters = set()

        # revealed phrase initially has underscores and spaces
        self.revealed_phrase = ['_' if letter.isalpha() else letter for letter in self.current_phrase]

        # tracks scores for each player- this is equivalent to accumulated "money" in the game
        self.scores = [0] * self.num_players 

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
            return True
        else:
            print("Incorrect solution.")
            return False

    def display_status(self):
        print("Phrase:", ' '.join(self.revealed_phrase))
        print(f"Scores: {', '.join([f'Player {i+1}: {score}' for i, score in enumerate(self.scores)])}")
        print(f"Guessed Letters: {', '.join(sorted(self.guessed_letters))}")

    def play_turn(self):
        print(f"-- Player {self.current_player + 1}'s Turn --")
        self.display_status()
        action = input("Choose an action - (S)pin the wheel, (B)uy a vowel, (P)uzzle solve: ").upper()
        if action == 'S':
            spin_result = self.spin_wheel()
            if spin_result == "Bankrupt":
                print("You went bankrupt!")
                self.scores[self.current_player] = 0
                self.bankrupt[self.current_player] = True
            elif spin_result == "Lose a Turn":
                print("You lost your turn!")
            else:
                guessed_letter = input("Guess a consonant: ").upper()
                if guessed_letter in VOWELS:
                    print("You cannot guess a vowel, you need to buy them.")
                    return
                if self.guess_letter(guessed_letter):
                    self.scores[self.current_player] += spin_result * self.current_phrase.count(guessed_letter)
                else:
                    print("Incorrect guess.")
        elif action == 'B':
            letter = input("Enter a vowel to buy: ").upper()
            self.buy_vowel(letter)
        elif action == 'P':
            guess = input("Enter your solution for the puzzle: ")
            if self.solve_puzzle(guess):
                return
        else:
            print("Invalid action. Please choose again.")
            self.play_turn()

    def is_solved(self):
        return '_' not in self.revealed_phrase

    def start_game(self):
        print("Welcome to Wheel of Fortune!")
        while not self.is_solved():
            self.play_turn()
            if self.is_solved():
                break
            self.current_player = (self.current_player + 1) % self.num_players
        if self.is_solved():
            print("You have solved the puzzle!")
            self.display_status()
        else:
            print("Game over.")

if __name__ == "__main__":
    game = WheelOfFortune(phrases, wheel_values)
    game.start_game()
