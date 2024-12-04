import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from players.player import Player
from data.phrases import phrases

MAX_PHRASE_LENGTH = max(len(phrase) for phrase in phrases)

class DQNPlayer(Player):
    def __init__(self, player_id, state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, target_update=10):
        super().__init__(player_id)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.batch_size = batch_size

        # replay buffer
        self.memory = deque(maxlen=buffer_size)

        # neural nets
        self.policy_net = self._build_model()
        self.target_net = self._build_model()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # training steps counter
        self.steps = 0

    def _build_model(self):

        # the model accepts the encoded state vector as input 
        # and outputs the Q-values for each action
        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def pad_phrase(self, phrase):
        phrase = phrase.upper()
        if len(phrase) < MAX_PHRASE_LENGTH:
            return phrase + "_" * (MAX_PHRASE_LENGTH - len(phrase))
        return phrase[:MAX_PHRASE_LENGTH]

    def encode_state(self, game, player_id):
        # encode revealed phrase
        revealed_phrase = ''.join(game.revealed_phrase)
        padded_phrase = self.pad_phrase(revealed_phrase)
        phrase_vector = [26 if char == "_" else ord(char) - ord("A") for char in padded_phrase]

        phrase_one_hot = np.zeros((MAX_PHRASE_LENGTH, 27))
        for i, idx in enumerate(phrase_vector):
            phrase_one_hot[i, idx] = 1
        phrase_one_hot = phrase_one_hot.flatten()

        # normalize player's score
        score = game.scores[player_id] / 100.0

        # one-hot encode guessed letters
        guessed_vector = np.zeros(26)
        for letter in game.guessed_letters:
            guessed_vector[ord(letter) - ord("A")] = 1

        # combine everything into a single state vector
        state_vector = np.concatenate([phrase_one_hot, [score], guessed_vector])
        return state_vector


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

    def spin(self, game):
        # TODO- spin the wheel and return a reward
        return 0

    def buy_vowel(self, game):
        # TODO- buy a vowel and return a reward
        return 0

    def solve_puzzle(self, game):
        # TODO- solve the puzzle and return a reward
        return 0

    def take_turn(self, game):
        # get current state and choose an action
        state = self.encode_state(game, self.player_id)
        action = self.act(game)

        # Perform action
        if action == 0:
            reward = self.spin(game)
        elif action == 1:
            reward = self.buy_vowel(game)
        elif action == 2:
            reward = self.solve_puzzle(game)
        else:
            reward = -1  # Invalid action penalty

        # Get next state
        next_state = self.encode_state(game, self.player_id)
        done = game.is_solved()

        # Remember experience
        self.remember(state, action, reward, next_state, done)

        # Replay and update target network
        self.replay()
        self.update_target_network()

        self.steps += 1
