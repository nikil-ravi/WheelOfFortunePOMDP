from dotenv import load_dotenv
load_dotenv()


from naive_player import NaivePlayer
from openai import OpenAI
import random
from constants import *
import os
from data.words import words

class LLMPlayer(NaivePlayer):
    def __init__(self, player_id):
        super().__init__(player_id)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def choose_consonant(self):
        available_consonants = [c for c in CONSONANTS if c not in self.game.guessed_letters]
        if not available_consonants:
            return None

        revealed_phrase = ''.join(self.game.revealed_phrase)
        guessed_letters = ', '.join(sorted(self.game.guessed_letters))

        prompt = f"""
        Wheel of Fortune Game:
        - Revealed Phrase: '{revealed_phrase}'
        - Guessed Letters: {guessed_letters}
        
        Choose ONLY ONE consonant from this list: {', '.join(available_consonants)}
        Respond with the single consonant letter, followed by a brief explanation of your choice.
        Format: <LETTER>: <explanation>
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Updated to use GPT-4o
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.5,
            )

            raw_response = response.choices[0].message.content.strip()
            
            # Extract the consonant and explanation
            parts = raw_response.split(':', 1)
            suggested_consonant = parts[0].strip().upper()
            explanation = parts[1].strip() if len(parts) > 1 else "No explanation provided."

            if suggested_consonant in available_consonants:
                print(f"AI suggested consonant: {suggested_consonant}")
                print(f"Reasoning: {explanation}")
                return suggested_consonant
            else:
                print(f"AI suggested invalid consonant: {suggested_consonant}")
                print(f"Full response: {raw_response}")
                print("Choosing randomly.")
                return random.choice(available_consonants)

        except Exception as e:
            print(f"Error in API call: {e}. Choosing randomly.")
            return random.choice(available_consonants)
        
    def choose_vowel(self):
        available_vowels = [v for v in VOWELS if v not in self.game.guessed_letters]
        if not available_vowels:
            return None

        revealed_phrase = ''.join(self.game.revealed_phrase)
        guessed_letters = ', '.join(sorted(self.game.guessed_letters))

        prompt = f"""
        Wheel of Fortune Game:
        - Revealed Phrase: '{revealed_phrase}'
        - Guessed Letters: {guessed_letters}
        
        Choose ONLY ONE vowel from this list: {', '.join(available_vowels)}
        Respond with the single vowel letter, followed by a brief explanation of your choice.
        Format: <LETTER>: <explanation>
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.5,
            )

            raw_response = response.choices[0].message.content.strip()
            
            # Extract the vowel and explanation
            parts = raw_response.split(':', 1)
            suggested_vowel = parts[0].strip().upper()
            explanation = parts[1].strip() if len(parts) > 1 else "No explanation provided."

            if suggested_vowel in available_vowels:
                print(f"AI suggested vowel: {suggested_vowel}")
                print(f"Reasoning: {explanation}")
                return suggested_vowel
            else:
                print(f"AI suggested invalid vowel: {suggested_vowel}")
                print(f"Full response: {raw_response}")
                print("Choosing randomly.")
                return random.choice(available_vowels)

        except Exception as e:
            print(f"Error in API call: {e}. Choosing randomly.")
            return random.choice(available_vowels)
        
    def choose_action(self):
        player_score = self.game.scores[self.player_id]
        possible_solutions = self.get_possible_solutions()
        
        if len(possible_solutions) == 1:
            return 'P'
        elif player_score < COST_OF_VOWEL:
            return 'S'
        else:
            return random.choice(['B', 'S'])
