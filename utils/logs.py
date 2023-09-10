import numpy as np

def hour_min_sec(duration):
	"""
	A function to convert a duration in seconds to hours, mins, secs.

	Parameters
	----------
	duration: float
		time in seconds.

	Returns
	------
	tuple of floats
		hours, minutes, and seconds of duration.
	
	"""
    hours = int(np.floor(duration / 3600))
    mins = int(np.floor(duration%3600 / 60))
    secs = duration % 60
    
    return hours, mins, secs

def print_time_elapsed(start):
	"""
	A function to print time elapsed based on a start time.

	Parameters
	----------
	start: float
		start time to reference from.

	Returns
	------
	None
	
	"""
    import time
    duration = time.time() - start
    hours, mins, secs = hour_min_sec(duration)
    print(f"{hours} hours, {mins} minutes, and {secs :0.1f} seconds")