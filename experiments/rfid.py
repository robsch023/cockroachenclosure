"""
Basic read/number acquire codes on an RFID reader.

The end goal is to identify RFID and provide an actuation signal to the camera

"""

from mfrc522 import MFRC522

def connect():
    """
    connect establises single active connection to the rfid chip
    
    """
    reader = MFRC22()
    status = None
    
    while status != reader.MI_OK:
        (status, TagType) = reader.Request(reader.PICC_REQIDL)

    (status, uid) = reader.Anticoll()
    if status != reader.MI_OK:
        return
    
    reader.SelectTag(uid)
    print("MFRCC connected.")
    print(f"Status:{status} | UID: {uid}")
    return(reader, uid)

def read_sector(reader, uid, trailer_blocks, blocks):
    print(f"Try read")
