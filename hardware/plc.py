"""This script contains definition of PLC class"""

from pymodbus.client.sync import ModbusTcpClient

class PLC():
    """PLC Class"""
    
    def __init__(self, ip_address: str):
        """Constructor method
        Args:
            ip_address: IP address of PLC to be connected
        """
        # Define attribute for PLC IP Address
        self.plc_ip = ip_address
        
    def write_words(self, values: tuple):
        """Write PLC Registers through Modbus TCP Communication
        Args:
            values: tuple of values to be written to register
        """
        # Define Modbus TCP Client and connect to it
        client = ModbusTcpClient(self.plc_ip)
        client.connect()
        
        # Write value to registers: %MW0, %MW1, and so on
        print("Writing value to PLC registers...")
        for reg, val in enumerate(values):
            client.write_registers(reg, int(val))
        print("Writing operation done!")
        
        # Close connection
        client.close()
        
    def write_bits(self, bit_value: str):
        """Write PLC Coils through Modbus TCP Communication
        Args:
            bit_value: binary value in string format
        """
        # Define Modbus TCP Client and connect to it
        client = ModbusTcpClient(self.plc_ip)
        client.connect()
        
        # Write value to coils: %M0, %M1, and so on
        print("Writing value to PLC coils...")
        for coil, bit in enumerate(bit_value):
            client.write_coils(coil, [int(bit)])
        print("Writing operation done!")
            
        # Close connection
        client.close()