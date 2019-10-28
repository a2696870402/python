from socket import *
import time

# 1 定义ip 和 端口
HOST,POST="192.168.56.1",8000

Buffer_size=1024
ADDR=(HOST,POST)

tcpServerSocket=socket(AF_INET,SOCK_STREAM)

tcpServerSocket.bind(ADDR)

tcpServerSocket.listen(5)

print("连接成功!,等待客户端发消息")

while True:

    tcpClientSocket, addr = tcpServerSocket.accept()
    while True:

        data=tcpClientSocket.recv(Buffer_size).decode()
        if not data:
            break
        if data == "1":
            tcpClientSocket.send("发的是1".encode())
        elif data == "2":
            tcpClientSocket.send("发的是2".encode())
        print("data=", data)
    tcpClientSocket.close()

tcpServerSocket.close()


