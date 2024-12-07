import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Script to train and evaluate a Q-learning agent on Wheel of Fortune with a train-test split.

Usage:
    python train_q_learning.py [options]

Example:
    python train_q_learning.py --num_episodes 10000 --epsilon 0.1
"""

import argparse
import pickle
import random

from data.phrases import phrases
from data.words import words
from data.wheel_values import wheel_values
from players.q_learning_player import QLearningPlayer
from wof import WheelOfFortune

def train_q_learning_player(
    train_phrases,
    wheel_values,
    num_episodes=5000,
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1,
    save_path="q_table.pkl",
    save_interval=100
):
    # Initialize game and player using training phrases only
    game = WheelOfFortune(
        train_phrases,
        wheel_values,
        num_players=1,
        player_class=lambda pid: QLearningPlayer(pid, alpha=alpha, gamma=gamma, epsilon=epsilon)
    )
    player = game.players[0]  # QLearningPlayer

    for episode in range(1, num_episodes + 1):
        game.reset_game()
        while not game.is_solved():
            game.play_turn()
            if game.is_solved():
                break
            game.current_player = (game.current_player + 1) % len(game.players)

        # Print out the episode number each time
        print(f"Episode {episode} completed.")

        # Save progress periodically
        if episode % save_interval == 0:
            print(f"Episode {episode}/{num_episodes} completed. Saving Q-table...")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(player.q_table, f)
            print(f"Q-table saved to {save_path}.")

    # Save final Q-table
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(player.q_table, f)
    print("Training completed.")
    print(f"Final Q-table saved to {save_path}.")

def evaluate_q_learning_player(test_phrases, wheel_values, q_table_path, num_games=100):
    # Create a QLearningPlayer with epsilon=0 (greedy) and no learning to evaluate
    player = QLearningPlayer(player_id=0, alpha=0, gamma=0.9, epsilon=0)

    # Load the trained Q-table
    with open(q_table_path, "rb") as f:
        player.q_table = pickle.load(f)

    solved_count = 0
    total_score = 0

    # Create a separate game instance for testing with the test phrase set
    test_game = WheelOfFortune(test_phrases, wheel_values, num_players=1, player_class=lambda pid: player)

    for _ in range(num_games):
        test_game.reset_game()

        while not test_game.is_solved():
            test_game.play_turn()
            if test_game.is_solved():
                solved_count += 1
                break
            test_game.current_player = (test_game.current_player + 1) % len(test_game.players)

        total_score += test_game.scores[0]

    avg_score = total_score / num_games
    solve_rate = (solved_count / num_games) * 100
    print("Evaluation on test set:")
    print(f"  Average Score: {avg_score:.2f}")
    print(f"  Solved Puzzles: {solved_count}/{num_games} ({solve_rate:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Q-learning agent on Wheel of Fortune with a train-test split.")
    parser.add_argument("--num_episodes", type=int, default=5000, help="Number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor for future rewards.")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Initial epsilon for exploration.")
    parser.add_argument("--save_path", type=str, default="models/q_table.pkl", help="Path to save the Q-table.")
    parser.add_argument("--save_interval", type=int, default=100, help="Save Q-table every N episodes.")
    parser.add_argument("--test_games", type=int, default=100, help="Number of games to evaluate on test set.")

    args = parser.parse_args()

    # Shuffle and split phrases into train (90%) and test (10%)
    all_phrases = words[:]
    random.shuffle(all_phrases)
    split_index = int(0.9 * len(all_phrases))
    train_phrases = all_phrases[:split_index]
    test_phrases = all_phrases[split_index:]

    print("Starting training of Q-learning agent...")
    print(f"Training phrases: {len(train_phrases)}")
    print(f"Testing phrases: {len(test_phrases)}")

    # Train on the training set
    train_q_learning_player(
        train_phrases,
        wheel_values,
        num_episodes=args.num_episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        save_path=args.save_path,
        save_interval=args.save_interval
    )

    # Evaluate on the test set
    evaluate_q_learning_player(
        test_phrases,
        wheel_values,
        q_table_path=args.save_path,
        num_games=args.test_games
    )
