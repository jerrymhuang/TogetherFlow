using UnityEngine;
using System.Linq;

public class LocomotiveAgent : MonoBehaviour
{

    private GameObject beacon;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        transform.localPosition = Vector3.right * Random.Range(-4f, 4f) + 
                                  Vector3.forward * Random.Range(-5f, 5f);
        beacon = GameObject.FindGameObjectWithTag("Beacon");
    }


    // Update is called once per frame
    void FixedUpdate()
    {
        WalkTo(beacon);
    }


    /// <summary>
    /// Implements the 2D Euler-Maruyama scheme for random walk with varying drift.
    /// </summary>
    /// <param name="beacon"></param>
    void WalkTo(GameObject beacon, float baseDrift = 1f, float decay = 0.01f, float scale = 1f)
    {
        Vector3 position = transform.position;
        float distanceToBeacon = Distance2D(position, beacon.transform.position);
        float drift = baseDrift * distanceToBeacon * decay;
        float displacement = drift * Time.deltaTime;
        position.x += displacement + scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();
        position.z += displacement + scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();
        transform.position = position;
    }


    float Distance2D(Vector3 start, Vector3 end)
    {
        float d;
        float x = start.x - end.x;
        float z = start.z - end.z;
        d = x * x + z * z;
        return d;
    }
}
