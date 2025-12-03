
# --- Imports ---

import difflib

from utilities.global_definitions import message

# --- Main function ---

def accuracy_calculator(recieved_message):

    """
    Calculates the reciever accuracy by comparing sent vs. recieved message.

    Arguments:
        "recieved_message" (str)

    Returns:
        "accuracy_percentage" (float)

    """

    if recieved_message:

        sequenceMatcher = difflib.SequenceMatcher(None, message, recieved_message)

        accuracy_percentage = round(sequenceMatcher.ratio() * 100)

        return accuracy_percentage
    
    else:

        return 0