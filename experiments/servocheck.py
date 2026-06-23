"""
servo function test file

should jog servo between positions of open and closed (512, 200)
"""

from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

PORT     = "/dev/ttyUSB0"
BAUD     = 1_000_000
ID       = 1
PROTOCOL = 1.0

port   = PortHandler(PORT)
packet = PacketHandler(PROTOCOL)

if not port.openPort():
    print("FAIL: could not open port — check /dev/ttyUSB0 exists (ls /dev/ttyUSB*)")
elif not port.setBaudRate(BAUD):
    print("FAIL: could not set baud rate")
else:
    print(f"Port open on {PORT} at {BAUD} baud")
    pos, result, error = packet.read2ByteTxRx(port, ID, 36)  # addr 36 = present position
    if result != COMM_SUCCESS:
        print(f"FAIL: no response — {packet.getTxRxResult(result)}")
        print("Check: servo powered? correct ID? correct baud?")
    elif error != 0:
        print(f"WARN: hardware error — {packet.getRxPacketError(error)}")
    else:
        print(f"OK: servo ID {ID} present position = {pos}  ({pos / 1023 * 300:.1f}°)")
        
    input("Press Enter to jog to position 512 (~150°)...")
    packet.write2ByteTxRx(port, ID, 30, 512)  # addr 30 = goal position
    import time; time.sleep(2)
    packet.write2ByteTxRx(port, ID, 30, pos)  # return to original position
    print("Jog complete")

    port.closePort()

