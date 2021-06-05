
import numpy as np
import socket

host, port = "127.0.0.1", 25001
data = "true"



s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


s.connect((host, port))
data = "happy"
while True:
    s.sendall(data.encode("utf-8"))
