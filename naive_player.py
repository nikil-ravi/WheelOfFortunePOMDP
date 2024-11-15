# from player import Player
# import random
# from constants import *
# from data.words import words


# ## write a naive implemmentaion of choose_solution, choose_consonant, choose_vowel and choose_action to override the Player class
# class NaivePlayer(Player):
#     def __init__(self, player_id):
#         super().__init__(player_id)
#         self.previous_guesses = set()
#         self.guessed_solutions = set()
    
#     def choose_action(self):
#         # placeholder- choose a random action
#         return random.choice(['S', 'B', 'P'])

#     def choose_vowel(self):
#         available_vowels = [v for v in VOWELS if v not in self.previous_guesses]
#         if not available_vowels:
#             return None  # All vowels have been guessed
#         vowel = random.choice(available_vowels)
#         self.previous_guesses.add(vowel)
#         return vowel

#     def choose_solution(self, game):
#         # Get the current state of the revealed phrase
#         revealed_phrase = ''.join(game.revealed_phrase)
#         print("Trying to choose solution for revealed phrase:", revealed_phrase)
                
#        # Find all phrases that match the revealed phrase and the previous guesses
#         possible_solutions = []

#         for phrase in game.phrases:
#             # Check if the solution has been guessed already
#             if phrase in self.guessed_solutions:
#                 continue

#             # Check if the lengths match
#             if len(phrase) != len(revealed_phrase):
#                 continue

#             # Check if the phrase matches the revealed pattern
#             matches_pattern = True
#             for c1, c2 in zip(revealed_phrase, phrase):
#                 if c1 != '_' and c1.upper() != c2.upper():
#                     matches_pattern = False
#                     break
#             if not matches_pattern:
#                 continue

#             # Check if the phrase is consistent with previous guesses
#             consistent_with_guesses = True
#             for letter in self.previous_guesses:
#                 if (letter.upper() in phrase.upper()) != (letter.upper() in revealed_phrase.upper()):
#                     consistent_with_guesses = False
#                     break
#             if not consistent_with_guesses:
#                 continue

#             # If we've made it here, this phrase is a possible solution
#             possible_solutions.append(phrase)

#         print("Possible solutions:", possible_solutions)
#         if possible_solutions:
#             chosen_solution = random.choice(possible_solutions)
#             self.guessed_solutions.add(chosen_solution)  # Add the chosen solution to guessed solutions
#             return chosen_solution
#         else:
#             return ""

    
#     def choose_consonant(self):
#         available_consonants = [c for c in CONSONANTS if c not in self.previous_guesses]
#         if not available_consonants:
#             return None  # All consonants have been guessed
#         consonant = random.choice(available_consonants)
#         self.previous_guesses.add(consonant)
#         return consonant
 
#     def take_turn(self, game):
#         action = self.choose_action()
#         print(f"Player {self.player_id + 1} chose action: {action}")
#         if action == 'S':
#             self.spin(game)
#         elif action == 'B':
#             letter = self.choose_vowel()
#             if letter:
#                 game.buy_vowel(letter)
#             else:
#                 print("No vowels left to guess.")
#         elif action == 'P':
#             guess = self.choose_solution(game)
#             print(f"Player {self.player_id + 1} guessed solution: {guess}")
#             game.solve_puzzle(guess)
#         else:
#             print("Invalid action. Please choose again.")
#             self.take_turn(game)

#     def guess(self, game, spin_result):
#         guessed_letter = self.choose_consonant()
#         if guessed_letter in VOWELS:
#             print("You cannot guess a vowel, you need to buy them.")
#             return
#         if guessed_letter is None:
#             print("No consonants left to guess.")
#             return
#         if game.guess_letter(guessed_letter):
#             points = spin_result * game.current_phrase.count(guessed_letter)
#             game.scores[self.player_id] += points
#         else:
#             print("Incorrect guess.")

    



from player import Player
import random
from constants import *
from data.words import words

class NaivePlayer(Player):
    def __init__(self, player_id):
        super().__init__(player_id)
    
    def choose_action(self):
        # placeholder- choose a random action
        return random.choice(['S', 'B', 'P'])

    def choose_vowel(self):
        available_vowels = [v for v in VOWELS if v not in self.game.guessed_letters]
        return random.choice(available_vowels) if available_vowels else None

    def choose_solution(self):
        revealed_phrase = ''.join(self.game.revealed_phrase)
        print("Trying to choose solution for revealed phrase:", revealed_phrase)
                
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

            consistent_with_guesses = True
            for letter in self.game.guessed_letters:
                if (letter.upper() in phrase.upper()) != (letter.upper() in revealed_phrase.upper()):
                    consistent_with_guesses = False
                    break
            if not consistent_with_guesses:
                continue

            possible_solutions.append(phrase)

        # print("Possible solutions:", possible_solutions)
        return random.choice(possible_solutions) if possible_solutions else ""

    def choose_consonant(self):
        available_consonants = [c for c in CONSONANTS if c not in self.game.guessed_letters]
        return random.choice(available_consonants) if available_consonants else None
 
    def take_turn(self, game):
        self.game = game  # Ensure the player has the latest game state
        action = self.choose_action()
        print(f"Player {self.player_id + 1} chose action: {action}")
        if action == 'S':
            self.spin(game)
        elif action == 'B':
            letter = self.choose_vowel()
            if letter:
                game.buy_vowel(letter)
            else:
                print("No vowels left to guess. Spinning instead.")
                self.spin(game)
        elif action == 'P':
            guess = self.choose_solution()
            print(f"Player {self.player_id + 1} guessed solution: {guess}")
            game.solve_puzzle(guess)

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
        if guessed_letter is None:
            print("No consonants left to guess. Ending turn.")
            return
        if game.guess_letter(guessed_letter):
            points = spin_result * game.current_phrase.count(guessed_letter)
            game.scores[self.player_id] += points
        else:
            print("Incorrect guess.")