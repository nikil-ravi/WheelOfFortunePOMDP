from constants import VOWELS, CONSONANTS
import random
import numpy as np
from collections import Counter
from naive_player import NaivePlayer

COST_OF_VOWEL = 250

file_path = 'data/popular.txt'
lines = []
with open(file_path, 'r') as file:
    for line in file:
        lines.append(line.strip().upper())

def find_char_positions(char, string):
    # Find all positions of char in string
    return [i for i, c in enumerate(string) if c == char]

class InfoGainPlayer(NaivePlayer):
    def __init__(self, player_id):
        self.player_id = player_id
        self.bankrupt = False
        self.candidates = lines.copy()

    def take_turn(self, game):
        self.candidates = self.find_candidate_words(game, game.revealed_phrase)
        action = action = self.choose_action(game)
        if action == 'S':
            self.spin(game)
        elif action == 'B':
            letter = self.choose_vowel(game)
            game.buy_vowel(letter)
        elif action == 'P':
            guess = self.choose_solution()
            game.solve_puzzle(guess)
        else:
            print("Invalid action. Please choose again.")
            self.take_turn(game)
        return action

    def find_candidate_words(self, game, pattern):
        # Find words from all_words that fit pattern.
        matching_words = self.candidates.copy()
        matching_words = [test_word for test_word in matching_words if len(pattern)==len(test_word)]
        for i in range(len(pattern)):
            if pattern[i]!='_':
                matching_words = [test_word for test_word in matching_words if test_word[i]==pattern[i]]
        for char in game.guessed_letters:
            matching_words = [test_word for test_word in matching_words if find_char_positions(char, pattern)==find_char_positions(char, test_word)]
        return matching_words

    def choose_action(self, game):
        player_score = game.scores[self.player_id]
        if len(self.candidates)==1:
            return 'P'
        elif player_score < COST_OF_VOWEL:
            return 'S'
        else:
            return random.choice(['B', 'S'])
    
    def reveal_letter_in_pattern(self, pattern, letter, true_word):
        return ''.join([
                letter if true_word[i] == letter else pattern[i]
                for i in range(len(true_word))
            ])
    
    # def choose_vowel(self, game):
    #     curr_entropy = np.log(len(self.candidates))
    #     S = len(self.candidates)
    #     information_gain = [0 for _ in range(len(VOWELS))]
    #     for v in range(len(VOWELS)):
    #         if v in game.guessed_letters:
    #             information_gain[v] = 0
    #             continue
    #         letter = VOWELS[v]
    #         new_candidates = [self.reveal_letter_in_pattern(game.revealed_phrase, letter, true_word) for true_word in self.candidates]
    #         counts = Counter(new_candidates)
    #         new_entropy = np.sum([i*np.log(i)/S for i in counts.values()])
    #         information_gain[v] = (curr_entropy-new_entropy)/curr_entropy
    #     print("Information gain with vowels:", information_gain)
    #     return VOWELS[np.argmax(information_gain)]

    def choose_vowel(self, game):
        VOWELS_NOT_REVEALED = [letter for letter in VOWELS if letter not in game.guessed_letters]
        if len(VOWELS_NOT_REVEALED)==0:
            return random.choice(VOWELS)
        curr_entropy = np.log(len(self.candidates))
        S = len(self.candidates)
        information_gain = np.zeros(len(VOWELS_NOT_REVEALED))
        for v in range(len(VOWELS_NOT_REVEALED)):
            letter = VOWELS_NOT_REVEALED[v]
            new_candidates = [self.reveal_letter_in_pattern(game.revealed_phrase, letter, true_word) for true_word in self.candidates]
            counts = Counter(new_candidates)
            new_entropy = np.sum([i*np.log(i)/S for i in counts.values()])
            if curr_entropy==0:
                information_gain[v] = 0
            else:
                information_gain[v] = (curr_entropy-new_entropy)/curr_entropy
        print("Information gain with vowels:", information_gain)
        return VOWELS_NOT_REVEALED[np.argmax(information_gain)]

    def choose_solution(self):
        # placeholder- this can be overridden by subclasses
        # TODO: figure out what a good baseline should be for this- maybe a random phrase?
        return random.choice(self.candidates)

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
        guessed_letter = self.choose_consonant(game)
        if guessed_letter in VOWELS:
            print("You cannot guess a vowel, you need to buy them.")
            return
        if guessed_letter in game.guessed_letters:
            print(f"{guessed_letter} has already been guessed.")
            return
        if game.guess_letter(guessed_letter):
            points = spin_result * game.current_phrase.count(guessed_letter)
            game.scores[self.player_id] += points
        else:
            print("Incorrect guess.")

    def find_letter_probs(self, game):
        # Find probability of each letter being present in the phrase based on self.candidates.
        # If letter is already in guessed_letters, probability of 0 is assigned.
        letter_probs = np.zeros(26)
        for word in self.candidates:
            for letter in set(word):
                if letter not in game.guessed_letters:
                    letter_probs[ord(letter)-ord('A')]+=1
        letter_probs/=len(self.candidates)
        return letter_probs

    def find_best_consonant(self, game):
        # Choose consonant that is present in most words from candidates
        letter_probs = self.find_letter_probs(game)
        for vowel in VOWELS:
            letter_probs[ord(vowel)-ord('A')]=0
        print("Probability of consonant =", np.max(letter_probs))
        return chr(ord('A')+np.argmax(letter_probs))

    def choose_consonant(self, game):
        # placeholder- choose a random consonant
        return self.find_best_consonant(game)