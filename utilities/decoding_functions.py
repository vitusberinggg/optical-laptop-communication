
# --- Imports ---

from utilities import color_functions

from utilities.global_definitions import number_of_rows, number_of_columns
from utilities.decoding_functions_v3_1 import bits_to_message

# --- Functions ---

bits = [[[]]]

def decode_bitgrid(frame, frame_bit=0, add_frame=False, recall=False, end_frame=False):
    global bits

    h, w = frame.shape[:2]
    bit_cell_height = h / number_of_rows
    bit_cell_width  = w / number_of_columns

    if add_frame:

        # ensure list for this frame
        while len(bits) <= frame_bit:
            bits.append([])

        for row in range(number_of_rows):

            # ensure row list exists
            while len(bits[frame_bit]) <= row:
                bits[frame_bit].append([])

            for column in range(number_of_columns):

                # ensure column exists
                while len(bits[frame_bit][row]) <= column:
                    bits[frame_bit][row].append(None)

                # extract ROI
                y0 = int(row * bit_cell_height)
                y1 = int(y0 + bit_cell_height)
                x0 = int(column * bit_cell_width)
                x1 = int(x0 + bit_cell_width)
                cell = frame[y0:y1, x0:x1]

                if end_frame:
                    bit = color_functions.tracker.end_bit(row, column)

                    # Ensure safe bit (string "0" / "1")
                    if bit not in ["0", "1"]:
                        bit = "0"

                    bits[frame_bit][row][column] = bit

                else:
                    color_functions.tracker.add_frame(cell, row, column)

        return None

    if recall:

        collected_bytes = []
        current_byte = []

        for f in range(frame_bit):      # each finalized frame
            for row in range(number_of_rows):
                for column in range(number_of_columns):

                    value = bits[f][row][column]

                    # safety: convert None → "0"
                    if value is None:
                        value = "0"

                    current_byte.append(value)

                    if len(current_byte) == 8:
                        collected_bytes.append(current_byte)
                        current_byte = []

        print(f"Decoded {len(collected_bytes)} bytes from {frame_bit} frames.")
        for i, byte_bits in enumerate(collected_bytes):
            byte_str = "".join(str(b) for b in byte_bits)
            print(f"Byte {i}: {byte_str} (char: '{chr(int(byte_str,2))}')")
        return bits_to_message(collected_bytes)

    return None