using System.Collections.Generic;
using UnityEngine;

public class Flock {
    public List<Boid> boids;
    public Flock() {
        boids = new List<Boid>();
    }

    public void Run() {
        foreach (var boid in boids) {
            boid.Run(boids.ToArray());  // Passing the entire list of boids to each boid individually
        }
    }

    public void AddBoid(Boid b) => boids.Add(b);
}