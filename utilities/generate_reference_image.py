
# --- Imports ---

import numpy as np

from global_definitions import reference_image_seed, sender_output_height, sender_output_width

# --- Main function ---

def generate_reference_image():

    """
    Generates a reference image by putting the image seed into a bit generator.

    Arguments:
        None

    Returns:
        "reference_image": The reference image.

    """

    bit_generator = np.random.RandomState(reference_image_seed)
    reference_image = bit_generator.randint(0, 256, (sender_output_height, sender_output_width), dtype = np.uint8)

    return reference_image