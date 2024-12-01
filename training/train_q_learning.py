import pickle
from wheel_of_fortune import WheelOfFortune
from q_learning_player import QLearningPlayer  # Assuming this is the updated QLearningPlayer class

# Number of training episodes
NUM_EPISODES = 10000
SAVE_INTERVAL = 100  # Save the Q-table every 100 episodes
Q_TABLE_SAVE_PATH = "q_table.pkl"

# Function to train the QLearningPlayer
def train_q_learning_player(phrases, wheel_values, save_path=Q_TABLE_SAVE_PATH):
    # Initialize the game with a single QLearningPlayer
    game = WheelOfFortune(phrases, wheel_values, num_players=1, player_class=QLearningPlayer)
    player = game.players[0]  # Assuming a single player

    for episode in range(1, NUM_EPISODES + 1):
        game.reset_game()
        solved = False

        # Run a single episode
        while not game.is_solved():
            game.play_turn()
            if game.is_solved():
                solved = True
                break
            game.current_player = (game.current_player + 1) % len(game.players)

        # Print progress
        if episode % SAVE_INTERVAL == 0:
            print(f"Episode {episode}/{NUM_EPISODES} completed. Saving Q-table...")

            # Save the Q-table
            with open(save_path, "wb") as f:
                pickle.dump(player.q_table, f)

            print(f"Q-table saved to {save_path}.")

    # Save final Q-table
    with open(save_path, "wb") as f:
        pickle.dump(player.q_table, f)
    print("Training completed. Final Q-table saved.")

if __name__ == "__main__":
    from data.phrases import phrases
    from data.wheel_values import wheel_values

    train_q_learning_player(phrases, wheel_values)
