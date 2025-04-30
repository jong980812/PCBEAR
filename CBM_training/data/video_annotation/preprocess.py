# Let's create a Python script to process the given input text in a file, remove commas, and create a new file with line breaks

# Function to process the text
def process_text_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    with open(output_file, 'w') as file:
        for line in lines:
            # Remove commas and replace with line breaks
            processed_line = line.replace(',', '\n').strip() + '\n'
            file.write(processed_line)

# Example file paths (you can change these to actual file paths)
input_file = '/data/jong980812/project/Video-CBM/data/kintics400_ver1.txt'
output_file = '/data/jong980812/project/Video-CBM/data/kintics400_ver2.txt'

# Call the function to process the file
process_text_file(input_file, output_file)


