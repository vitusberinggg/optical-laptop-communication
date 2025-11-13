
# --- Main function ---

def bits_to_message(bits):

    """
    Converts a string of bits into a readable message.
    
    Arguments:
        "bits" (str): A string representing the bits to be converted.
        
    Returns:
        str: A string representing the decoded message.
        
    """

    characters = []

    for bit_index in range(0, len(bits), 8): # For each byte (8 bits) in the bit string:

        byte = bits[bit_index:bit_index + 8] # Extract the byte using slicing

        if len(byte) < 8: # If the length of the byte is less than 8 bits:
            continue # Skip it

        characters.append(chr(int(byte, 2))) # Convert the byte to a character and append it to the list

    return "".join(characters) # Return the decoded message by joining the list of characters into a string