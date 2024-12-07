import random
from data.phrases import phrases
from data.wheel_values import wheel_values
from constants import *
from players.player import Player
from players.naive_player import NaivePlayer
from players.llm_player import LLMPlayer
from data.words import words
from llm_player import LLMPlayer
from wof import WheelOfFortune

if __name__ == "__main__":
<<<<<<< HEAD:main.py
    game = WheelOfFortune(words, wheel_values, player_class=LLMPlayer)
=======
    game = WheelOfFortune(words, wheel_values, player_class= LLMPlayer)
>>>>>>> main:wof_word.py
    game.start_game()
