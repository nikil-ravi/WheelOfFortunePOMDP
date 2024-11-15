import random
from data.phrases import phrases
from data.wheel_values import wheel_values
from constants import *
from player import Player
from naive_player import NaivePlayer
from data.words import words
from wof import WheelOfFortune

if __name__ == "__main__":
    game = WheelOfFortune(words, wheel_values, player_class= NaivePlayer)
    game.start_game()
