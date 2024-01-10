"""Module for Modbus Communication with PLC"""

from pymodbus.client.sync import ModbusTcpClient
from pyModbusTCP.server import DataBank, DeviceIdentification

def prepare_server():
    """Prepare server properties"""
    identity = DeviceIdentification(
        vendor_name=b"Lab. Schneider DTETI UGM",
        product_code=b"MCD",
        major_minor_revision=b"1.0",
        vendor_url=b"https://github.com/javanese-programmer/conveyor-object-detection",
        product_name=b"McDetector",
        model_name=b"McDetector-1",
        user_application_name=b"Conveyor Object Detection"
    )
    databank = DataBank()
    return databank, identity

def server_set_di(databank: DataBank, bit_value: str):
    """Set a value to Discretes Input of a server
    Args:
        databank: a DataBank object that store data
        bit_value: binary value in string format
    """
    # Set value to discretes inputs
    print("Updating discretes inputs...")
    for addr, bit in enumerate(bit_value):
        databank.set_discrete_inputs(addr, [int(bit)])
    print("Discretes Inputs updated!")
    

def server_set_ir(databank: DataBank, values: tuple):
    """Set a value to Input Registers of a server
    Args:
        databank: a DataBank object that store data
        values: tuple of values to be written to register
    """
    # Set value to Input registers
    print("Updating input registers...")
    for addr, val in enumerate(values):
        databank.set_input_registers(addr, [int(val)])
    print("Input registers updated!")


class PLC():
    """PLC Client Class"""
    
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