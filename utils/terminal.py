"""Utility functions to print/prompt data in terminal"""

def print_det_time(count, ir_sns, cmr, start, delay):
    """Print detection time in to terminal/console
    Args:
      count: detection count
      ir_sns: ir sensor detection time
      cmr: camera detection time
      start: start detection time
      delay: delay between detection
    """
    rounded_irs = round((ir_sns - start), 3)
    rounded_cmr = round((cmr - start), 3)
    rounded_dly = round(delay, 3)
    print(f"Detection Session: {count}")
    print(f"IR sensor detection at {rounded_irs} second")
    print(f"Camera detection at {rounded_cmr} second")
    print(f"Delay between detections: {rounded_dly} second")


def print_detected(name, score):
    """Print messages if object is detected
    Args:
      name: name of detected object
      score: probability score
    """
    print(f"The object is {name}")
    print(f"The probability is {score}")


def print_undetected(count):
    """Print messages if nothing detected
    Args:
      count: detection count
    """
    print(f"Detection Session: {count}")
    print("Object is not detected!")


def print_dimension(dimension, score):
    """print messages about detected dimension
    Args:
      dimension: detected dimension of object
      score: probability score of an object
    """
    print("Dimension:")
    print(f"The height is {dimension[0]} cm")
    print(f"The width is {dimension[1]} cm")
    print(f"The size/area is {dimension[2]} cm2")
    print(f"Probability: {score}")


def print_color(color, score):
    """Print messages if object is detected
    Args:
      color: BGR tuple of detected object
      score: probability score
    """
    print(f"The BGR color is {color}")
    print(f"The probability is {score}")
    
def prompt_label():
    """Prompt user to input the true label"""
    true_label = str(input("True label of the object: "))
    atuple = tuple(true_label.strip('()').split(','))
    if len(atuple) == 1:
        return atuple[0]
    else:
        return (int(atuple[0]), int(atuple[1]), int(atuple[2]))
    
def prompt_type():
    """Prompt user to input the true label"""
    print("For traditional method, detection type has to be shape or color")
    det_type = str(input("Detection Type: "))
    print("")
    
    return det_type