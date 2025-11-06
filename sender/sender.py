
# --- Imports ---

import time
import tkinter as tk

# --- Definitions ---

binary_duration = 0.3
delimiter_duration = 0.5

# --- Helper functions ---

def show_color(color, duration = binary_duration):
    canvas.config(background = color)
    root.update()
    time.sleep(duration)

# --- Main function ---

def send_message(message):

    for character in message:
        bits = format(ord(character), "08b")

        for bit in bits:
            show_color("white" if bit == "1" else "black")

        show_color("red", delimiter_duration)

# --- GUI setup ---

root = tk.Tk()
root.attributes("-fullscreen", True)
canvas = tk.Canvas(root, highlightthickness = 0)
canvas.pack(fill = "both", expand = True)

# --- Main execution ---

time.sleep(2)
send_message("HELLO")