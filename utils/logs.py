import numpy as np

def hour_min_sec(duration):
    hours = int(np.floor(duration / 3600))
    mins = int(np.floor(duration%3600 / 60))
    secs = duration % 60
    
    return hours, mins, secs

def print_time_elapsed(start):
    import time
    duration = time.time() - start
    hours, mins, secs = hour_min_sec(duration)
    print(f"{hours} hours, {mins} minutes, and {secs :0.1f} seconds")