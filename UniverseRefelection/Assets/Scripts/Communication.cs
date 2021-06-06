using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;


public class Communication : MonoBehaviour {
    private const int listenPort = 11000;
    private UdpClient listener;
    private IPEndPoint groupEP;

    private void Start() {
        StartListener();
    }

    private void Update() {
        try {
            // Debug.Log("Waiting for broadcast");
            byte[] bytes = listener.Receive(ref groupEP);

            Debug.Log($"Received broadcast from {groupEP} :");
            Debug.Log($" {Encoding.ASCII.GetString(bytes, 0, bytes.Length)}");
        }
        catch (SocketException e) {
            Debug.LogError(e);
        }
    }

    private void StartListener() {
        listener = new UdpClient(listenPort);
        groupEP = new IPEndPoint(IPAddress.Any, listenPort);
        listener.Client.Blocking = false;
    }

    private void OnDisable() => listener.Close();
    private void OnDestroy() => listener.Close();

    private void OnApplicationQuit() => listener.Close();
}