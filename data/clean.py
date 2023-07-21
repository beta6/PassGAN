import re, sys

def filter_ascii(binary_data):
    # Filter only ASCII characters from the binary data
    ascii_bytes = [b for b in binary_data if 0x00 <= b <= 0x7F]
    
    # Create a cleaned binary string using the filtered bytes
    cleaned_binary_data = bytes(ascii_bytes)
    
    return cleaned_binary_data.decode("ascii")



def clean_file(input_file, output_file):
    pattern = re.compile(r'[^a-zA-Z0-9!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~\n\r\t]')
    with open(input_file, 'rb') as in_file, open(output_file, 'w') as out_file:
        for line in in_file:
            line=filter_ascii(line)
            if not pattern.search(line) and len(line)<12:
                out_file.write(line)

# Example usage
if len(sys.argv)>=3:
    clean_file(sys.argv[1], sys.argv[2])
else:
    print("%s <input file> <output file>")

