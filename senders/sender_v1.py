
# --- Imports ---

import time
import tkinter as tk

# --- Definitions ---

binary_duration = 0.1
delimiter_duration = 0.1

# --- Helper functions ---

def update_screen_color(color, duration = binary_duration):

    """
    Updates the screen with the given color.

    Arguments:
        "color"
        "duration"

    Returns:
        None

    """

    canvas.config(background = color)
    root.update()
    time.sleep(duration)

# --- Main function ---

def send_message(message):

    """
    Sends the message by calling "update_screen_color".

    Arguments:
        "message"

    Returns:
        None
    
    """
    update_screen_color("green", binary_duration * 9)

    for character in message: # For each character in the message:

        bits = format(ord(character), "08b") # Convert the character to it's corresponding ASCII chain of 1's and 0's

        for bit in bits: # For each bit:

            if bit == "1":
                update_screen_color("white")

            else:
                update_screen_color("black")

            update_screen_color("blue", binary_duration)

        update_screen_color("red", delimiter_duration)

# --- GUI setup ---

root = tk.Tk()
root.attributes("-fullscreen", True)

canvas = tk.Canvas(root, highlightthickness = 0)
canvas.pack(fill = "both", expand = True)

# --- Main execution ---

time.sleep(1)
send_message("HELLO")
