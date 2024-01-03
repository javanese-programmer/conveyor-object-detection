"""Python module to count, time, and calculate detection"""

import time
from utils import terminal

class Flags():
    """Flags state class"""
    
    def __init__(self):
        """Constructor method"""
        # Define flags attributes
        self.print_message = False
        self.detected_ir = False
        self.detected_cmr = False
        
    def reverse_cmr(self):
        """Reverse the state of camera"""
        self.detected_cmr = not(self.detected_cmr)
    
    def reverse_ir(self):
        """Reverse the state of IR"""
        self.detected_ir = not(self.detected_ir)
        
    def reverse_msg(self):
        """Reverse the state of message print"""
        self.print_message = not(self.print_message)
        

class Counter():
    """Detection counter class"""
    
    def __init__(self):
        """Constructor method"""
        # Attributes to count frame and detection
        self.frame_count = 0
        self.det_count = 0
        self.img_count = 0
        
    def frame_up(self):
        """Count up frame"""
        self.frame_count += 1
        
    def det_up(self):
        """Count up detection"""
        self.det_count += 1
        
    def img_up(self):
        """Count up captured images"""
        self.img_count += 1
        

class Timer():
    """Detection timer class"""
    
    def __init__(self):
        """Constructor method"""
        # Attributes to calculate detection time
        self.det_start = time.time()
        self.start_time = time.time()
        self.end_time = 0
        self.old_cmr_time = 0
        self.cmr_time = 0
        self.old_ir_time = 0
        self.ir_time = 0
        
        # Attribute to calculate Modbus communication time
        self.register_time = 0
        self.coil_time = 0
        
    def restart_ir(self):
        """Restart IR detection timer"""
        self.ir_time = time.time()
        self.old_ir_time = self.ir_time
        
    def restart_cmr(self):
        """Restart camera detection timer"""
        self.cmr_time = time.time()
        self.old_cmr_time = self.cmr_time
        
    def start_reg(self):
        """Start write registers operation timer"""
        self.register_time = time.time()
    
    def start_coil(self):
        """Start write coils operation timer"""
        self.coil_time = time.time()
        

class Calculator(Timer, Counter):
    """Detection calculator class"""
    
    def __init__(self):
        """Constructor method"""
        # Inherit attribute from base class
        Timer.__init__(self)
        Counter.__init__(self)
        
        # Attributes to calculate delay and FPS
        self.fps_avg_frame_count = 10
        self.delay = 0
        self.fps = 0
        
        # Attribute to calculate latency of Modbus comm
        self.reg_latency = 0
        self.coil_latency = 0
        
        # Attributes to collect all data
        self.fps_list = []
        self.delay_list = []
        self.reg_latency_list = []
        self.coil_latency_list = []
    
    def calculate_delay(self):
        """Calculate delay between ir and camera detection"""
        self.delay = self.cmr_time - self.ir_time
        
    def calc_reg_latency(self):
        """Calculate the latency to write data to PLC registers"""
        self.reg_latency = time.time() - self.register_time
        
    def calc_coil_latency(self):
        """Calculate the latency to write data to PLC registers"""
        self.coil_latency = time.time() - self.coil_time
        
    def calculate_fps(self):
        """Calculate the frame rate"""
        if self.frame_count % self.fps_avg_frame_count == 0:
            self.end_time = time.time()
            self.fps = self.fps_avg_frame_count / (self.end_time - self.start_time)
            self.start_time = time.time()
        else:
            pass
            
    def update_data(self, detected_cmr: bool):
        """Update all data
        Args:
            detected_cmr: Whether object is detected on camera or not
        """
        self.fps_list.append(round(self.fps,3))
        self.reg_latency_list.append(round(self.reg_latency, 6))
        self.coil_latency_list.append(round(self.coil_latency, 6))
        
        # If object detected on camera, update delay value
        if detected_cmr:
            self.delay_list.append(round(self.delay,3))
        else:
            self.delay_list.append(0)
            
        
    def print_data(self):
        """Print all collected data"""
        terminal.print_det_time(self.det_count, self.ir_time, self.cmr_time,
                                self.det_start, self.delay)