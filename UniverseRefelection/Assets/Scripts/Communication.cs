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

    [SerializeField] private FlockingBoids[] FlockingBoidsArray;

    private void Start() {
        StartListener();
    }

    private void Update() {
        try {
            // Debug.Log("Waiting for broadcast");
            byte[] bytes = listener.Receive(ref groupEP);
            var receivedMessage = Encoding.ASCII.GetString(bytes, 0, bytes.Length);
            switch (receivedMessage) {
                
                case "happy":
                {
                    // boids will be less spaced out and higher instances
                    foreach (var flock in FlockingBoidsArray)
                    {
                        if (flock.isMainFlock) break;
                        flock.maximumDistance = flock.minSeparation / 2.0f;
                        flock.desiredSeparation = flock.minSeparation;
                    }
                    break;
                }
                case "sad":
                {
                    // boids more spaced out and fewer instances
                    foreach (var flock in FlockingBoidsArray)
                    {
                        if (flock.isMainFlock) break;
                        flock.maximumDistance = flock.maxSeparation / 2.0f;
                        flock.desiredSeparation = flock.maxSeparation;
                    }
                    break;
                }
                case "neutral":
                {
                    // boids more spaced out and fewer instances
                    foreach (var flock in FlockingBoidsArray)
                    {
                        if (flock.isMainFlock) break;   
                    }
                    break;
                }

            }
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