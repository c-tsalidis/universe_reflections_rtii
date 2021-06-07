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
    public float minSpeed = 1.0f, maxMaxSpeed = 5.0f;
    public float maxforce = 0.05f; // Maximum steering force
    public float maximumDistance = 100.0f;
    public float desiredSeparation = 10.0f;
    public float minSeparation = 5.0f, maxSeparation = 20.0f;
    
    public bool isMainFlock = false;

    // private variables
    private Transform _cameraTransform;
    
    
    private void Start() {
        _flock = new Flock();
        // Add an initial set of boids into the system
        for (int i = 0; i < numberOfBoids; i++) {
            var b = Instantiate(boidPrefab, transform.position, Quaternion.identity);
            // var b = new Boid(Screen.width / 2.0f, Screen.height / 2.0f, 0);
            // if(i == 0) Camera.main.transform.SetParent(b.transform);
            var boid = b.GetComponent<Boid>();
            boid.SetUp(Screen.width / 2.0f, Screen.height / 2.0f, 0);
            _flock.AddBoid(boid);
        }

        _cameraTransform = Camera.main.transform;
    }

    private void Update() {
        _flock.UpdateBoids(r, maxspeed, maxforce, maximumDistance, desiredSeparation);

        // Camera.main.transform.position = Vector3.Lerp(Camera.main.transform.position, _flock.boids[15].transform.position, 1);

        if(!isMainFlock) return;
        _cameraTransform.position = _flock.AveragePosition();
         _cameraTransform.LookAt(Vector3.zero);
        // _cameraTransform.rotation = Quaternion.Slerp(_cameraTransform.rotation, _flock.boids[_flock.boids.Count / 2].transform.rotation, 50);
        // _cameraTransform.rotation = Quaternion.Lerp(Camera.main.transform.rotation, _flock.boids[15].transform.rotation, 100);
        //_cameraTransform.rotation = _flock.boids[_flock.boids.Count / 2].transform.rotation;
    }

        private void OnMouseDrag() {
        var b = Instantiate(boidPrefab);
        var boid = b.GetComponent<Boid>();
        boid.SetUp(Input.mousePosition.x, Input.mousePosition.y, 0);
        _flock.AddBoid(boid);
    }
}

