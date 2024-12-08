import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import random
import time
import numpy as np
from data.words import words
from data.wheel_values import wheel_values
from players.q_learning_player import QLearningPlayer
from wof import WheelOfFortune

class CustomWheelOfFortune(WheelOfFortune):
    def __init__(self, phrases, wheel_values, num_players=2, player_class=QLearningPlayer, mode="Training", episode_number=1):
        self.mode = mode
        self.episode_number = episode_number
        super().__init__(phrases, wheel_values, num_players=num_players, player_class=player_class)

    def play_turn(self):
        self.turn_count += 1
        current_player = self.players[self.current_player]
        print(f"-- {self.mode} Episode {self.episode_number}, Player {current_player.player_id + 1}'s Turn (Turn count: {self.turn_count}) --")
        self.display_status()
        current_player.take_turn(self)

def train_qtable_player(
    train_phrases,
    wheel_values,
    num_episodes=5000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.1,
    save_path="qtable_model.pkl"
):
    total_score = 0
    start_time = time.time()
    # Initialize outside loop since we re-instantiate the game each episode (to set episode_number)
    for episode in range(1, num_episodes + 1):
        game = CustomWheelOfFortune(
            train_phrases,
            wheel_values,
            num_players=1,
            player_class=lambda pid: QLearningPlayer(
                pid, 
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon_start,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min
            ),
            mode="Training",
            episode_number=episode
        )

        player = game.players[0]
        # Set current epsilon (if needed) directly from player
        player.epsilon = max(player.epsilon * (player.epsilon_decay if player.epsilon_decay else 1.0), player.epsilon_min)

        game.reset_game()
        while not game.is_solved():
            game.play_turn()
            game.current_player = (game.current_player + 1) % len(game.players)

        player.end_game(game)
        total_score += game.scores[0]

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed. Epsilon: {player.epsilon:.4f}")

        epsilon_start = player.epsilon  # Update epsilon_start for next episode

    end_time = time.time()
    total_time = end_time - start_time
    average_time_per_episode = total_time / num_episodes
    average_score = total_score / num_episodes

    # Save Q-table from the last player
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(player.q_table, f)
    print(f"Training completed. Q-table saved to {save_path}.")
    print(f"Average time per training episode: {average_time_per_episode:.4f} seconds")
    print(f"Average final score after training episodes: {average_score:.2f}")

def evaluate_qtable_player(eval_phrases, wheel_values, model_path="qtable_model.pkl", num_games=1000):
    with open(model_path, 'rb') as f:
        q_table = pickle.load(f)

    total_turns = 0
    solved_count = 0
    total_score = 0

    start_time = time.time()
    for i in range(num_games):
        player = QLearningPlayer(0, epsilon=0)
        player.q_table = q_table
        game = CustomWheelOfFortune(
            eval_phrases, 
            wheel_values, 
            num_players=1,
            player_class=lambda pid: player,
            mode="Evaluation",
            episode_number=i+1
        )

        game.reset_game()
        num_turns = 0
        while not game.is_solved():
            game.play_turn()
            num_turns += 1
            if game.is_solved():
                solved_count += 1
                break
            game.current_player = (game.current_player + 1) % len(game.players)

        total_turns += num_turns
        total_score += game.scores[0]

    end_time = time.time()

    avg_turns = total_turns / num_games
    avg_score = total_score / num_games
    total_eval_time = end_time - start_time
    avg_time_per_game = total_eval_time / num_games

    print(f"Average number of turns per evaluation game: {avg_turns:.2f}")
    print(f"Solved Games: {solved_count}/{num_games}")
    print(f"Average final score per evaluation game: {avg_score:.2f}")
    print(f"Average time per evaluation game: {avg_time_per_game:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Q-table agent on Wheel of Fortune.")
    parser.add_argument("--num_episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards.")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Initial epsilon for exploration.")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate per episode.")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon value.")
    parser.add_argument("--save_path", type=str, default="models/qtable_model.pkl", help="Path to save the trained Q-table.")
    
    args = parser.parse_args()

    # Implement 90-10 split
    random.shuffle(words)
    split_index = int(0.9 * len(words))
    train_phrases = words[:split_index]
    eval_phrases = words[split_index:]

    print("Starting training of Q-table agent...")
    train_qtable_player(
        train_phrases,
        wheel_values,
        num_episodes=args.num_episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        save_path=args.save_path
    )

    print("\nEvaluating trained Q-table agent...")
    evaluate_qtable_player(eval_phrases, wheel_values, model_path=args.save_path)
