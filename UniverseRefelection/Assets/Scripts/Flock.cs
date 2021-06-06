using System.Collections.Generic;
using UnityEngine;

public class Flock {
    public List<Boid> boids;

    public Flock() {
        boids = new List<Boid>();
    }

    public void UpdateBoids(float r, float maxSpeed, float maxSteeringForce, float maximumDistance, float desiredSeparation) {
        foreach (var boid in boids) {
            boid.r = r;
            boid.maxspeed = maxSpeed;
            boid.maxforce = maxSteeringForce;
            boid.maximumDistance = maximumDistance;
            boid.desiredSeparation = desiredSeparation;
            boid.Run(boids.ToArray()); // Passing the entire list of boids to each boid individually
        }
    }

    public void AddBoid(Boid b) => boids.Add(b);

    public Vector3 AveragePosition() {
        var average = Vector3.zero;
        foreach (var b in boids) {
            average = Calculate.Add(average, b.transform.position);
        }

        average = Calculate.Divide(boids.Count, average);
        return average;
    }
    
    public Quaternion AverageRotation() {
        var average = Vector3.zero;
        foreach (var b in boids) {
            average = Calculate.Add(average, b.transform.rotation.eulerAngles);
        }

        average = Calculate.Divide(boids.Count, average);
        return Quaternion.Euler(average);
    }
}