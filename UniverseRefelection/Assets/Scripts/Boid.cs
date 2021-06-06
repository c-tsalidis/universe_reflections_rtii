using UnityEngine;
using System;
using Unity.Mathematics;

public class Boid: MonoBehaviour {
    public Vector3 acceleration;
    private Vector3 velocity;
    private Vector3 position;
    public float r = 3.0f;
    public float maxspeed = 3.0f; // maximum speed
    public float maxforce = 0.05f; // Maximum steering force
    [SerializeField] private float maximumDistance = 100.0f;

    public void SetUp(float x, float y, float z) {
        acceleration = new Vector3(0, 0, 0);
        velocity = new Vector3(UnityEngine.Random.Range(-1, 1), UnityEngine.Random.Range(-1, 1), 0);
        // position = new Vector3(x, y, z);
    }

    public void Run(Boid [] boids) {
        flock(boids);
        BoidUpdate();
        // this.borders();
        Render();
    }

    private void Render() {
        // update position and rotation of boid
        // var theta = Quaternion.LookRotation(velocity) + Mathf.PI;
        transform.rotation = Quaternion.LookRotation(velocity, Vector3.up);
        transform.position = new Vector3(position.x, position.y, position.z);
    }

    private void BoidUpdate() {
        // Update velocity
        velocity = Calculate.Add(acceleration, velocity);
        // Limit speed
        velocity = Calculate.Limit(velocity, maxspeed);
        position = Calculate.Add(velocity, position);
        // Reset acceleration to 0 each cycle
        acceleration = Calculate.Multiply(0, acceleration);
    }

    private void flock(Boid [] boids) {
        var sep = this.Separate(boids);   // Separation
        var ali = this.Align(boids);      // Alignment
        var coh = this.Cohesion(boids);   // Cohesion
        // Arbitrarily weight these forces
        sep = Calculate.Multiply(1.5f, sep);
        ali = Calculate.Multiply(1.0f, ali);
        coh = Calculate.Multiply(1.0f, coh);
        // Add the force vectors to acceleration
        ApplyForce(sep);
        ApplyForce(ali);
        ApplyForce(coh);
    }

    public void ApplyForce(Vector3 force) {
        // We could add mass here if we want A = F / M
        acceleration = Calculate.Add(acceleration, force);
    }

    private Vector3 Cohesion(Boid[] boids) {
        var neighbordist = 50;
        var sum = Vector3.zero;   // Start with empty vector to accumulate all locations
        var count = 0;
        for (int i = 0; i < boids.Length; i++) {
            var d = Vector3.Distance(position, boids[i].position);
            if ((d > 0) && (d < neighbordist)) {
                sum = Calculate.Add(boids[i].position, sum); // Add location
                count++;
            }
        }
        if (count > 0) {
            sum = Calculate.Divide(count, sum);
            return this.Seek(sum);  // Steer towards the location
        } else {
            return Vector3.zero;
        }
    }

    private Vector3 Separate(Boid[] boids) {
        var desiredseparation = 25.0f;
        var steer = new Vector3(0, 0, 0);
        var count = 0;
        // For every boid in the system, check if it's too close
        for (int i = 0; i < boids.Length; i++) {
            var d = Vector3.Distance(position, boids[i].position);
            // If the distance is greater than 0 and less than an arbitrary amount (0 when you are yourself)
            if ((d > 0) && (d < desiredseparation)) {
                // Calculate vector pointing away from neighbor
                var diff = Calculate.Subtract(position, boids[i].position);
                diff.Normalize();
                diff = Calculate.Divide(d, diff);        // Weight by distance
                steer = Calculate.Add(steer, diff);
                count++;            // Keep track of how many
            }
        }
        // Average -- divide by how many
        if (count > 0) {
            steer = Calculate.Divide(count, steer);
        }

        // As long as the vector is greater than 0
        if (steer.magnitude > 0) {
            // Implement Reynolds: Steering = Desired - Velocity
            steer.Normalize();
            steer = Calculate.Multiply(maxspeed, steer);
            steer = Calculate.Subtract(steer, velocity);
            steer = Calculate.Limit(steer, maxforce);
        }
        return steer;
    }

    public Vector3 Seek(Vector3 target) {
        if (Vector3.Distance(target, Vector3.zero) > maximumDistance) {
            target = Vector3.zero;
        }
        var desired = Calculate.Subtract(target, position);  // A vector pointing from the location to the target
        // Normalize desired and scale to maximum speed
        desired.Normalize();
        desired = Calculate.Multiply(maxspeed, desired);
        // Steering = Desired minus Velocity
        var steer = Calculate.Subtract(desired, velocity);
        steer = Calculate.Limit(steer, maxforce);  // Limit to maximum steering force
        return steer;
    }

    public Vector3 Align(Boid [] boids) {
        var neighbordist = 50;
        var sum = Vector3.zero;
        var count = 0;
        for (int i = 0; i < boids.Length; i++) {
            var d = Vector3.Distance(position, boids[i].position);
            if ((d > 0) && (d < neighbordist)) {
                sum = Calculate.Add(sum, boids[i].velocity);
                count++;
            }
        }
        if (count > 0) {
            sum = Calculate.Divide(count, sum);
            sum.Normalize();
            sum = Calculate.Multiply(maxspeed, sum);
            var steer = Calculate.Subtract(sum, velocity);
            steer = Calculate.Limit(steer, maxforce);
            return steer;
        }

        return Vector3.zero;
    }
}
