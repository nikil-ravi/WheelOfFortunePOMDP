import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pickle
from data.phrases import phrases
from data.wheel_values import wheel_values
from wof import WheelOfFortune
from players.q_learning_player import QLearningPlayer

def evaluate_q_learning_player(phrases, wheel_values, q_table_path, num_episodes=1):
    # Create a QLearningPlayer with epsilon=0 (greedy) and no learning updates
    player = QLearningPlayer(player_id=0, alpha=0, gamma=0.9, epsilon=0)

    # Load the trained Q-table
    with open(q_table_path, "rb") as f:
        player.q_table = pickle.load(f)

    solved_count = 0
    total_score = 0
    total_turns = 0

    for episode in range(1, num_episodes + 1):
        game = WheelOfFortune(phrases, wheel_values, num_players=1, player_class=lambda pid: player)
        game.reset_game()
        turns = 0

        while not game.is_solved():
            game.play_turn()
            turns += 1
            if game.is_solved():
                solved_count += 1
                break
            game.current_player = (game.current_player + 1) % len(game.players)

        total_score += game.scores[0]
        total_turns += turns

    print(f"Evaluation over {num_episodes} episodes:")
    print(f"  Solved puzzles: {solved_count}/{num_episodes} ({solved_count/num_episodes*100:.2f}%)")
    print(f"  Average Score: {total_score/num_episodes:.2f}")
    print(f"  Average Turns: {total_turns/num_episodes:.2f}")

if __name__ == "__main__":
    # Replace 'q_table.pkl' with your actual Q-table file path
    evaluate_q_learning_player(phrases, wheel_values, q_table_path="q_table.pkl", num_episodes=100)
