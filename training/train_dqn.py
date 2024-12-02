

import numpy as np
from data.phrases import phrases
from WheelOfFortunePOMDP.players.dqn_player import DQNPlayer

MAX_PHRASE_LENGTH = max(len(phrase) for phrase in phrases)

def train_dqn_player(phrases, wheel_values, state_size, action_size, num_episodes=5000, save_path="dqn_model.pth"):
    
    # initialize game and player
    game = WheelOfFortune(
        phrases, wheel_values, num_players=1,
        player_class=lambda pid: DQNPlayer(pid, state_size, action_size)
    )
    player = game.players[0]  # DQNPlayer is the only player

    for episode in range(1, num_episodes + 1):
        game.reset_game()
        done = False

        while not game.is_solved():
            game.play_turn()
            if game.is_solved():
                done = True
                break
            game.current_player = (game.current_player + 1) % len(game.players)

        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes} completed.")

    # save model
    torch.save(player.policy_net.state_dict(), save_path)
    print(f"Model saved to {save_path}.")
    print("Training completed.")


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

    # TODO
    # state_size = 26 + 1 + (27 * len(phrases[0]))  # guessed letters + score + revealed phrase
    # action_size = 3  # spin, buy vowel, solve

    # train_dqn_player(phrases, wheel_values, state_size, action_size)


