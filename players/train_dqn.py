"""
Script to train a DQN agent on Wheel of Fortune.

Usage:
    python train_dqn.py [options]

Example:
    python train_dqn.py --num_episodes 10000 --batch_size 128 --gamma 0.95

"""

import argparse
import os
import torch

import numpy as np
from data.phrases import phrases
from data.wheel_values import wheel_values
from dqn_player import DQNPlayer
from wof import WheelOfFortune
from tqdm import tqdm

MAX_PHRASE_LENGTH = 20 #max(len(phrase) for phrase in phrases)

file_path = 'data/popular.txt'
lines = []
with open(file_path, 'r') as file:
    for line in file:
        lines.append(line.strip().upper())

def train_dqn_player(
    phrases,
    wheel_values,
    state_size=7, # 27*MAX_PHRASE_LENGTH+2+26,
    action_size=3,
    hidden_size=24,
    training_mode=True,
    num_episodes=3000,
    batch_size=64,
    gamma=0.99,
    learning_rate=1e-3,
    epsilon_start=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.1,
    target_update=10,
    save_path="dqn_model.pth"
):
    
    # initialize game and player
    game = WheelOfFortune(
        phrases, 
        wheel_values, 
        num_players=1,
        player_class=lambda pid: DQNPlayer(
            pid, 
            state_size,
            action_size,
            hidden_size,
            training_mode,
            buffer_size=10000,
            batch_size=batch_size,
            gamma=gamma,
            lr=learning_rate,
            epsilon=epsilon_start,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            target_update=target_update,
        )
    )
    player = game.players[0]  # DQNPlayer is the only player
    num_turns_list = []

    for episode in tqdm(range(1, num_episodes + 1)):
        game.reset_game()
        player.candidates = lines.copy()
        done = False

        num_turns = 0
        while not game.is_solved() and num_turns < 100:
            game.play_turn()
            num_turns += 1
            if game.is_solved():
                done = True
                break
            game.current_player = (game.current_player + 1) % len(game.players)
        
        if done:
            num_turns_list.append(num_turns)
        else:
            num_turns_list.append(100)
        
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed.")
            print(f"Mean of last 100 num_turns: {np.mean(num_turns_list[-100:])}")

    # save model
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(player.policy_net.state_dict(), save_path)
    print(f"Training completed.")
    print(f"Model saved to {save_path}.")
    return num_turns_list

def evaluate_dqn_player(phrases, wheel_values, state_size=7, action_size=3, model_path="dqn_model.pth", num_games=10):
    
    # load the trained model
    player = DQNPlayer(0, state_size=state_size, action_size=action_size, training_mode = False, epsilon=0.0, epsilon_decay=0.0, epsilon_min=0.0)
    player.policy_net.load_state_dict(torch.load(model_path))
    player.policy_net.eval()

    game = WheelOfFortune(
        phrases, wheel_values, num_players=1,
        player_class=lambda pid: player
    )

    total_score = 0
    solved_count = 0
    num_turns_list = []
    for _ in tqdm(range(num_games)):
        game.reset_game()
        player.candidates = lines.copy()
        done = False

        num_turns = 0
        while not game.is_solved() and num_turns < 30:
            game.play_turn()
            num_turns += 1
            if game.is_solved():
                done = True
                solved_count += 1
                break
            game.current_player = (game.current_player + 1) % len(game.players)

        if done:
            num_turns_list.append(num_turns)
        else:
            num_turns_list.append(100)
        total_score += game.scores[0]

    avg_score = total_score / num_games
    print(f"Average Score: {avg_score}")
    print(f"Solved Games: {solved_count}/{num_games}")
    return num_turns_list, avg_score, solved_count

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a DQN agent on Wheel of Fortune.")
    parser.add_argument("--num_episodes", type=int, default=5000, help="Number of training episodes.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon for exploration.")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate per episode.")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon value.")
    parser.add_argument("--target_update", type=int, default=10, help="Frequency of target network updates.")
    parser.add_argument("--save_path", type=str, default="models/dqn_model.pth", help="Path to save the trained model.")
    
    args = parser.parse_args()

    max_phrase_length = max(len(phrase) for phrase in phrases)
    state_size = 26 + 1 + (27 * max_phrase_length)  # Guessed letters, score, revealed phrase
    action_size = 3  # Spin, buy vowel, solve

    print("Starting training of DQN agent...")

    # Train the DQN agent
    train_dqn_player(
        phrases,
        wheel_values,
        state_size,
        action_size,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_start=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        target_update=args.target_update,
        save_path=args.save_path
    )