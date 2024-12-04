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
from players.dqn_player import DQNPlayer
from wof import WheelOfFortune

MAX_PHRASE_LENGTH = max(len(phrase) for phrase in phrases)

def train_dqn_player(
    phrases,
    wheel_values,
    state_size,
    action_size,
    num_episodes=5000,
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

    for episode in range(1, num_episodes + 1):
        game.reset_game()
        done = False

        num_turns = 0
        while not game.is_solved() and num_turns < 10:
            game.play_turn()
            num_turns += 1
            if game.is_solved():
                done = True
                break
            game.current_player = (game.current_player + 1) % len(game.players)

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed.")

    # save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(player.policy_net.state_dict(), save_path)
    print(f"Training completed.")
    print(f"Model saved to {save_path}.")

def evaluate_dqn_player(phrases, wheel_values, state_size, action_size, model_path="dqn_model.pth", num_games=100):
    
    # load the trained model
    player = DQNPlayer(0, state_size, action_size)
    player.policy_net.load_state_dict(torch.load(model_path))
    player.policy_net.eval()

    game = WheelOfFortune(
        phrases, wheel_values, num_players=1,
        player_class=lambda pid: player
    )

    total_score = 0
    solved_count = 0

    for _ in range(num_games):
        game.reset_game()
        done = False

        while not game.is_solved():
            game.play_turn()
            if game.is_solved():
                done = True
                solved_count += 1
                break
            game.current_player = (game.current_player + 1) % len(game.players)

        total_score += game.scores[0]

    avg_score = total_score / num_games
    print(f"Average Score: {avg_score}")
    print(f"Solved Games: {solved_count}/{num_games}")

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

    


