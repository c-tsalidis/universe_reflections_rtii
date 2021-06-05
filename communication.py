import socket
import struct
import traceback
import logging
import time
import numpy as np


def sending_and_receiving():
    s = socket.socket()
    socket.setdefaulttimeout(None)
    print('socket created ')
    port = 60000
    s.bind(('127.0.0.1', port))  # local host
    s.listen(30)  # listening for connection for 30 sec?
    print('socket listening ... ')
    while True:
        try:
            c, addr = s.accept()  # when port connected
            bytes_received = c.recv(4000)  # received bytes
            array_received = np.frombuffer(bytes_received, dtype=np.float32)  # converting into float array

            # nn_output = return_prediction(array_received) #NN prediction (e.g. model.predict())
            #bytes_to_send = struct.pack('%sf' % len(array), *array)  # converting float to byte
            #c.sendall(array)  # sending back

            c.sendall(bytearray([0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1]))
            c.close()
            print(array_received)
        except Exception as e:
            logging.error(traceback.format_exc())
            print("error")
            c.sendall(bytearray([]))
            c.close()
            break


sending_and_receiving()
