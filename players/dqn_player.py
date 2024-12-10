import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
# from players.player import Player
from naive_player import NaivePlayer

from data.phrases import phrases

from constants import VOWELS, CONSONANTS
from collections import Counter
COST_OF_VOWEL = 250
file_path = 'data/popular.txt'
lines = []
with open(file_path, 'r') as file:
    for line in file:
        lines.append(line.strip().upper())

MAX_PHRASE_LENGTH = 20 #max(len(phrase) for phrase in phrases) ## 27*MAX_PHRASE_LENGTH+2+26

def find_char_positions(char, string):
    # Find all positions of char in string
        return [i for i, c in enumerate(string) if c == char]

class DQNPlayer(NaivePlayer):
    def __init__(self, player_id, state_size=7, action_size=3, hidden_size=24, training_mode = True, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, target_update=10):
        super().__init__(player_id)
        self.state_size = state_size
        self.action_size = action_size
        self.training_mode = training_mode
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.batch_size = batch_size
        self.bankrupt = False
        self.candidates = lines.copy()

        # replay buffer
        self.memory = deque(maxlen=buffer_size)

        # neural nets
        self.policy_net = self._build_model(hidden_size)
        self.target_net = self._build_model(hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # training steps counter
        self.steps = 0

    def _build_model(self, hidden_size=24):

        # the model accepts the encoded state vector as input 
        # and outputs the Q-values for each action
        model = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size)
        )
        return model

    def pad_phrase(self, phrase):
        phrase = phrase.upper()
        if len(phrase) < MAX_PHRASE_LENGTH:
            return phrase + "_" * (MAX_PHRASE_LENGTH - len(phrase))
        return phrase[:MAX_PHRASE_LENGTH]

    # def encode_state(self, game, player_id):
    #     # encode revealed phrase
    #     revealed_phrase = ''.join(game.revealed_phrase)
    #     padded_phrase = self.pad_phrase(revealed_phrase)
    #     phrase_vector = [26 if char == "_" else ord(char) - ord("A") for char in padded_phrase]

    #     phrase_one_hot = np.zeros((MAX_PHRASE_LENGTH, 27))
    #     for i, idx in enumerate(phrase_vector):
    #         phrase_one_hot[i, idx] = 1
    #     phrase_one_hot = phrase_one_hot.flatten()

    #     # normalize player's score
    #     score = game.scores[player_id] / 100.0

    #     # one-hot encode guessed letters
    #     guessed_vector = np.zeros(26)
    #     for letter in game.guessed_letters:
    #         guessed_vector[ord(letter) - ord("A")] = 1

    #     # combine everything into a single state vector
    #     state_vector = np.concatenate([phrase_one_hot, [score], [len(self.candidates)], guessed_vector])
    #     return state_vector
    
    def encode_state(self, game, player_id):
        num_revealed_chars = len([i for i in game.revealed_phrase if i!='_'])
        num_hidden_chars = len([i for i in game.revealed_phrase if i=='_'])
        num_guessed_letters = len(game.guessed_letters)
        entropy = np.log(len(self.candidates))
        score = game.scores[player_id] / 100.0
        can_solve_puzzle = 1 if len(self.candidates)==1 else 0
        _, vowel_info_gain = self.choose_vowel(game, print_info_gain=False)
        return np.array([num_revealed_chars, num_hidden_chars, num_guessed_letters, score, vowel_info_gain, entropy, can_solve_puzzle])


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, game):
        state = self.encode_state(game, self.player_id)
        
        # epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size)) # explore
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        print("Q VALUES:", q_values)
        if not self.training_mode:
            if len(self.candidates)==1:
                return 2
        return torch.argmax(q_values).item() # exploit

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # sample mini-batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * next_q_values

        # compute predicted Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # compute loss
        loss = nn.MSELoss()(q_values, targets)

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_network(self):
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


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

    def spin(self, game):
        spin_result = game.spin_wheel()
        if spin_result == "Bankrupt":
            print(f"Player {self.player_id + 1} went bankrupt!")
            game.scores[self.player_id] = 0
            self.bankrupt = True
        elif spin_result == "Lose a Turn":
            print(f"Player {self.player_id + 1} lost their turn!")
        else:
            self.guess_consonant(game, spin_result)

    def guess_consonant(self, game, spin_result):
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
    
    def reveal_letter_in_pattern(self, pattern, letter, true_word):
        return ''.join([
                letter if true_word[i] == letter else pattern[i]
                for i in range(len(true_word))
            ])

    def choose_vowel(self, game, print_info_gain=True):
        VOWELS_NOT_REVEALED = [letter for letter in VOWELS if letter not in game.guessed_letters]
        if len(VOWELS_NOT_REVEALED)==0:
            return random.choice(VOWELS), 0
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
        if print_info_gain:
            print("Information gain with vowels:", information_gain)
        return VOWELS_NOT_REVEALED[np.argmax(information_gain)], np.max(information_gain)
    
    def choose_solution(self):
        # placeholder- this can be overridden by subclasses
        # TODO: figure out what a good baseline should be for this- maybe a random phrase?
        return random.choice(self.candidates)

    def take_turn(self, game):
        # get current state and choose an action
        self.candidates = self.find_candidate_words(game, game.revealed_phrase)
        # print(self.candidates)
        state = self.encode_state(game, self.player_id)
        action = self.act(game)

        # Perform action
        if action == 0:
            self.spin(game)
            reward = 0

        elif action == 1:
            letter, _ = self.choose_vowel(game, print_info_gain=True)
            bought_vowel = game.buy_vowel(letter)
            if not bought_vowel:
                reward = -1
            else:
                reward = 0
        elif action == 2:
            guess = self.choose_solution()
            solved_bool = game.solve_puzzle(guess)
            if solved_bool:
                reward = 200/(game.total_turns+1)
            else:
                reward = -10
        else:
            reward = -1  # Invalid action penalty
        print("Reward = ", reward)

        # Get next state
        next_state = self.encode_state(game, self.player_id)
        done = game.is_solved()

        if self.training_mode:
            # Remember experience
            self.remember(state, action, reward, next_state, done)

            # Replay and update target network
            self.replay()
            self.update_target_network()

            self.steps += 1

        possible_actions = ['S', 'B', 'P']
        return possible_actions[action]