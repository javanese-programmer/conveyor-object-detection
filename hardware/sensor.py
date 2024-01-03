""" Script to define GPIO, sensor, and actuator """

from RPi import GPIO
from time import sleep

def prepare_gpio():
    """Function to prepare GPIO"""
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    
def clean_gpio():
    """Function to clean GPIO port"""
    GPIO.cleanup()

class Infrared():
    """ IR Sensor Class """
    
    def __init__(self, right_ir: int, left_ir: int):
        """ Contructor method
        Args:
            right_ir (int): Right IR Sensor Pin
            left_ir (int): Left IR Sensor pin
        """
        self.right_ir = right_ir
        self.left_ir = left_ir
        
        GPIO.setup(self.right_ir, GPIO.IN)
        GPIO.setup(self.left_ir, GPIO.IN)
        
    def read_sensor(self):
        """Method to detect object presence using IR"""
        right_read = GPIO.input(self.right_ir)
        left_read = GPIO.input(self.left_ir)
        return right_read, left_read


class Led():
    """ LED Class """
    
    def __init__(self, yellow, red, blue):
        """ Contructor method
        Args:
            yellow (int): Yellow LED Pin
            red (int): Red LED pin
            blue (int): Blue LED Pin
        """
        self.yellow = yellow
        self.red = red
        self.blue = blue
        
        GPIO.setup(self.yellow, GPIO.OUT)
        GPIO.setup(self.red, GPIO.OUT)
        GPIO.setup(self.blue, GPIO.OUT)
        
    def turn_on(self, index: int):
        """Method to turn-on an LED according to detected object
        Args:
            index: Index of detected object class 
        """
        if index == 0:
            GPIO.output(self.yellow, True)
            GPIO.output(self.red, False)
            GPIO.output(self.blue, False)
        elif index == 1:
            GPIO.output(self.yellow, False)
            GPIO.output(self.red, True)
            GPIO.output(self.blue, False)
        elif index == 2:
            GPIO.output(self.yellow, False)
            GPIO.output(self.red, False)
            GPIO.output(self.blue, True)
        else:
            GPIO.output(self.yellow, False)
            GPIO.output(self.red, False)
            GPIO.output(self.blue, False)
        
    def turn_off(self):
        """Method to turn-off all LED"""
        GPIO.output(self.yellow, False)
        GPIO.output(self.red, False)
        GPIO.output(self.blue, False)