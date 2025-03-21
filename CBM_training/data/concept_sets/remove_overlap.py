from collections import Counter

# Step 1: Read the file
input_file_path = '/data/jong980812/project/Video-CBM/data/concept_sets/word80k.txt'  # Replace this with your actual file path
output_file_path = '/data/jong980812/project/Video-CBM/data/concept_sets/word80k_noun_filtered.txt'  # This is where the unique words will be saved

with open(input_file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Step 2: Split the content into lines (words)
words = content.splitlines()

# Step 3: Count the total words and find duplicates
total_words = len(words)
word_counts = Counter(words)
duplicate_words = {word: count for word, count in word_counts.items() if count > 1}
unique_words = list(word_counts.keys())

# Step 4: Save the unique words into a new file
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write("\n".join(unique_words))

# Output results
print(f"Total words: {total_words}")
print(f"Duplicate words: {len(duplicate_words)}")
print(f"Unique words saved to {output_file_path}")