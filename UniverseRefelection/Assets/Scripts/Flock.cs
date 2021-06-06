using System.Collections.Generic;
using UnityEngine;

public class Flock {
    public List<Boid> boids;

    public Flock() {
        boids = new List<Boid>();
    }

    public void UpdateBoids(float r, float maxSpeed, float maxSteeringForce, float maximumDistance) {
        foreach (var boid in boids) {
            boid.r = r;
            boid.maxspeed = maxSpeed;
            boid.maxforce = maxSteeringForce;
            boid.maximumDistance = maximumDistance;
            boid.Run(boids.ToArray()); // Passing the entire list of boids to each boid individually
        }
    }

    public void AddBoid(Boid b) => boids.Add(b);
}