"""Utility functions to process array-like data and csv"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def collect_data(par_list: list):
    """Collecting data from detection.
    Args:
      par_list: List of recorded parameter.
    Return:
      par_arr: Numpy array of parameter
      par_count: Numpy array of counter
    """
    # Convert list to numpy array
    par_arr = np.array(par_list)
    # Create list of ratio between zero and non-zero values
    non_zero = list(par_arr[par_arr != 0])
    all_zero = list(par_arr[par_arr == 0])
    par_ratio = [len(non_zero), len(all_zero)]
    # Create array of non-zero values
    non_zero.insert(0, 0)
    par_arr = np.array(non_zero)
    # Create array to count number of detection
    par_count = np.arange(0, len(par_arr), 1)
    # Return numpy arrays
    return par_arr, par_count, par_ratio


def stack_array(list_a: list, list_b: list, *args):
    """Stack multiple list together and convert it to array.
    Args:
      list_a: the first list to be stacked
      list_b: the second list to be stacked
    Return:
      A numpy array stack of inputs
    """
    array_a = np.array(list_a)
    array_b = np.array(list_b)
    final_stack = np.stack((array_a, array_b), axis=-1)

    for other_list in args:
        array_c = np.array(other_list)
        # Checks if list contains another iterables
        try:
            inside_list_size = len(other_list[0])
        except TypeError:
            array_c = array_c.reshape(len(array_c), 1)
        else:
            if isinstance(other_list[0], str):
                array_c = array_c.reshape(len(array_c), 1)
            else:
                array_c = array_c.reshape(len(array_c), inside_list_size)

        final_stack = np.hstack((final_stack, array_c))

    return final_stack


def create_csv(stacked_arr, method:str, det_type: str, filename: str):
    """Create a CSV file from stacked array
    Args:
      stacked_arr: Array of stacked parameters.
      method: computer vision method (traditional or deeplearning)
      det_type: Type of detections (color, shape, or category)
      filename: Name of the CSV file.
    """
    # Define CSV Columns
    columns = ["Delay", "FPS", "Latency (Regs)", "Latency (Coils)", "Detected"]
    if method == 'deeplearning':
        columns.extend(["Probability"])
        if det_type == 'color':
            columns.extend(["Blue (Pred)", "Green (Pred)", "Red (Pred)",
                            "Blue (True)", "Green (True)", "Red (True)"])
        elif det_type == 'shape':
            columns.extend(["Height (Pred)", "Width (Pred)", "Size (Pred)",
                            "Height (True)", "Width (True)", "Size (True)"])
        else:
            columns.extend(["Prediction", "True Label"])
    
    elif method == 'traditional':
        if det_type == 'color':
            columns.extend(["Blue", "Green", "Red"])
        elif det_type == 'shape':
            columns.extend(["Area", "Points"])
        columns.extend(["Prediction", "Label"])
        
    # Create DataFrame and export to CSV
    recorded_data = pd.DataFrame(stacked_arr, columns=columns)
    recorded_data.to_csv(filename, index=False)


def reset_list(*args):
    """Clear the content of all lists."""
    for this_list in args:
        this_list.clear()


def update_list(all_list: list, all_values: list):
    """Update list with newest value
    Args:
      all_list: all list to be updated
      all_values: values to update list
    """
    for (i, this_list) in enumerate(all_list):
        this_list.append(all_values[i])