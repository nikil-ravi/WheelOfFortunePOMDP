from constants import VOWELS, CONSONANTS
import random

class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.bankrupt = False

    def take_turn(self, game):
        action = action = self.choose_action()
        print(f"Player {self.player_id + 1} chose action: {action}")
        if action == 'S':
            self.spin(game)
        elif action == 'B':
            letter = self.choose_vowel()
            game.buy_vowel(letter)
        elif action == 'P':
            guess = self.choose_solution()
            game.solve_puzzle(guess)
        else:
            print("Invalid action. Please choose again.")
            self.take_turn(game)

    def choose_action(self):
        # placeholder- choose a random action
        return random.choice(['S', 'B', 'P'])

    def choose_vowel(self):
        # placeholder- choose a random vowel
        return random.choice(VOWELS)

    def choose_solution(self):
        # placeholder- this can be overridden by subclasses
        # TODO: figure out what a good baseline should be for this- maybe a random phrase?
        return ""

    def spin(self, game):
        spin_result = game.spin_wheel()
        if spin_result == "Bankrupt":
            print(f"Player {self.player_id + 1} went bankrupt!")
            game.scores[self.player_id] = 0
            self.bankrupt = True
        elif spin_result == "Lose a Turn":
            print(f"Player {self.player_id + 1} lost their turn!")
        else:
            self.guess(game, spin_result)

    def guess(self, game, spin_result):
        guessed_letter = self.choose_consonant()
        if guessed_letter in VOWELS:
            print("You cannot guess a vowel, you need to buy them.")
            return
        if game.guess_letter(guessed_letter):
            points = spin_result * game.current_phrase.count(guessed_letter)
            game.scores[self.player_id] += points
        else:
            print("Incorrect guess.")

    def choose_consonant(self):
        # placeholder- choose a random consonant
        return random.choice(CONSONANTS)