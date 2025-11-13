
# --- Imports ---

import time
import numpy as np

from global_definitions import samples_per_frame, sample_space

# --- Main function ---

def capture_and_calculate_average_frame(camera):
    
    """
    Grabs multiple snapshots from the camera and averages them to reduce noise and ensure a fresh frame.

    Arguments:

    """
    
    snapshots = []
    
    for _ in range(samples_per_frame): # For each sample:
        
        for _ in range(2):
            camera.grab() # Grab two frames to ensure the latest frame is captured (mini buffer flush)
        
        read_was_successful, frame = camera.read() # Read the frame and the boolean indicating if the read was successful or not
        
        if not read_was_successful: # If the read wasn't successful:
            continue # Skip this iteration
        
        snapshots.append(frame.astype(np.float32)) # Add the snapshot (converted to float32 format) to the list of snapshots
        
        time.sleep(sample_space) # Pause briefly before the next sample

    if not snapshots: # If there aren't any snapshots:
        return None # End the function and return None (to indicate an error)

    averaged_frame = np.mean(snapshots, axis = 0) # Calculate the average of all snapshots along the first axis (pixel-by-pixel).
    
    return averaged_frame.astype(np.float32) # Return the final averaged frame in float32 format.