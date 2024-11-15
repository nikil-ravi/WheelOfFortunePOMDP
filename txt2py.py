# Script to convert words.txt to words.py

# Read words from words.txt
with open('data/words.txt', 'r') as file:
    words = [line.strip().upper() for line in file if line.strip()]

# Write words to words.py
with open('data/words.py', 'w') as file:
    file.write("words = [\n")
    for word in words:
        file.write(f"    '{word}',\n")
    file.write("]\n")

print(f"Converted {len(words)} words from words.txt to words.py")