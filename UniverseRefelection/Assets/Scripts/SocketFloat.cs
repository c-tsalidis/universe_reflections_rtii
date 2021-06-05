using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using UnityEngine;

public class SocketFloat : MonoBehaviour
{

    public string ip = "127.0.0.1";
    public int port = 60000;
    private Socket client;
    [SerializeField]
    private float[] dataOut, dataIn; //debugging


    private void Start()
    {
        SendAndReceive(dataOut);
        float[] prediction = ServerRequest(myArray);
    }

    /// <summary>
    /// Helper function for sending and receiving.
    /// </summary>
    /// <param name="dataOut">Data to send</param>
    /// <returns></returns>
    protected float[] ServerRequest(float[] dataOut)
    {
        //print("request");
        dataOut = dataOut; //debugging
        dataIn = SendAndReceive(dataOut); //debugging
        return dataIn;
    }

    /// <summary> 
    /// Send data to port, receive data from port.
    /// </summary>
    /// <param name="dataOut">Data to send</param>
    /// <returns></returns>
    private float[] SendAndReceive(float[] dataOut)
    {
        //initialize socket
        float[] floatsReceived;
        client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        client.Connect(ip, port);
        if (!client.Connected) {
            Debug.LogError("Connection Failed");
            return null; 
        }

        //convert floats to bytes, send to port
        var byteArray = new byte[dataOut.Length * 4];
        Buffer.BlockCopy(dataOut, 0, byteArray, 0, byteArray.Length);
        client.Send(byteArray);

        //allocate and receive bytes
        byte[] bytes = new byte[4000];
        int idxUsedBytes = client.Receive(bytes);
        //print(idxUsedBytes + " new bytes received.");
        print(idxUsedBytes);
        //convert bytes to floats
        floatsReceived = new float[idxUsedBytes/4];
        Buffer.BlockCopy(bytes, 0, floatsReceived, 0, idxUsedBytes);
        print(floatsReceived);
        client.Close();
        return floatsReceived;
    }

}
