using UnityEngine;
using UnityEngine.UIElements;

public class FlockingBoids : MonoBehaviour {
    [Header("Flock")]
    private Flock _flock;
    [SerializeField] private int numberOfBoids = 100;

    [Header("Boid details")]
    [SerializeField] private GameObject boidPrefab;
    public float r = 3.0f;
    public float maxspeed = 3.0f; // maximum speed
    public float maxforce = 0.05f; // Maximum steering force
    public float maximumDistance = 100.0f;
    
    private void Start() {
        _flock = new Flock();
        // Add an initial set of boids into the system
        for (int i = 0; i < numberOfBoids; i++) {
            var b = Instantiate(boidPrefab);
            // var b = new Boid(Screen.width / 2.0f, Screen.height / 2.0f, 0);
            var boid = b.GetComponent<Boid>();
            boid.SetUp(Screen.width / 2.0f, Screen.height / 2.0f, 0);
            _flock.AddBoid(boid);
        }
    }

    private void Update() => _flock.UpdateBoids(r, maxspeed, maxforce, maximumDistance);

        private void OnMouseDrag() {
        var b = Instantiate(boidPrefab);
        var boid = b.GetComponent<Boid>();
        boid.SetUp(Input.mousePosition.x, Input.mousePosition.y, 0);
        _flock.AddBoid(boid);
    }
}

