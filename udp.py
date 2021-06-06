'''
import socket

localIP     = "127.0.0.1"
localPort   = 11000
bufferSize  = 1024
msgFromServer = b"Hello UDP Client"
bytesToSend = str.encode(msgFromServer)

# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

# Bind to address and ip
#UDPServerSocket.bind((localIP, localPort)) 

print("UDP server up and listening")

UDPServerSocket.sendto(msgFromServer)

# Listen for incoming datagrams
while(True):
    bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
    message = bytesAddressPair[0]
    address = bytesAddressPair[1]
    clientMsg = "Message from Client:{}".format(message)
    clientIP  = "Client IP Address:{}".format(address)

    print(clientMsg)
    print(clientIP)
    # Sending a reply to client
    UDPServerSocket.sendto(bytesToSend, address)
'''


import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 11000
MESSAGE = b"Hello, World!"

print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
print("message: %s" % MESSAGE)
sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))