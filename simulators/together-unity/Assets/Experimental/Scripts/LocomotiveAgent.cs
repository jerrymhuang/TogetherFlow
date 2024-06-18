using UnityEngine;
using System.Linq;

public class LocomotiveAgent : Agent
{

    GameObject[] beacons;
    float attentionDistance;
    float distanceToBeacon;

    GameObject beacon;
    Vector3 position;
    Vector3 positionToBeacon;
    Vector3 direction;

    bool visualize = true;


    void Start()
    {
        transform.localPosition = Vector3.right * Random.Range(-4f, 4f) + 
                                  Vector3.forward * Random.Range(-5f, 5f);
        beacons = GameObject.FindGameObjectsWithTag("Beacon");

        // Sample attention distance individually
        attentionDistance = Random.Range(10f, 25f);
        
    }


    void FixedUpdate()
    {
        beacon = FindNearestBeacon();
        Debug.Log(beacon.name);

        distanceToBeacon = Distance2D(transform.position, beacon.transform.position);
        if (distanceToBeacon < attentionDistance) 
            AttendTo(beacon);
        else UnattendFrom(beacon);

        if (visualize) Visualize();
    }


    /// <summary>
    /// Implements the 2D Euler-Maruyama scheme for random walk with distance-dependent drift.
    /// </summary>
    /// <param name="beacon"></param>
    void AttendTo(GameObject beacon, float baseDrift = 0.1f, float scale = 0.1f)
    {

        position = transform.localPosition;
        direction = Vector3.Normalize(beacon.transform.position - transform.position);


        float drift = baseDrift * distanceToBeacon;
        float displacement = drift * Time.deltaTime;
        position.x += displacement * direction.x + 
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();
        position.z += displacement * direction.z + 
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();


        transform.localPosition = Bounded(position);
        transform.forward = Vector3.RotateTowards(transform.forward, beacon.transform.position, 0.01f, 0f);
    }


    /// <summary>
    /// Reorient to a random location within the room.
    /// </summary>
    /// <param name="beacon"></param>
    void UnattendFrom(GameObject beacon)
    {
        // TODO (still not quite right)
        Vector3 direction = Vector3.forward * Random.Range(-5f, 5f) + Vector3.right * Random.Range(-4f, 4f);
        transform.forward = Vector3.RotateTowards(transform.forward, direction, 0.01f, 0f);
    }


    /*
    Vector3 Bound(Vector3 position)
    {
        if (position.x < -4f) position.x = -4f + Random.Range(-0.01f, 0.01f);
        if (position.x > 4f) position.x = 4f + Random.Range(-0.01f, 0.01f);
        if (position.z < -5f) position.z = -5f + Random.Range(-0.01f, 0.01f);
        if (position.z > 5f) position.z = 5f + Random.Range(-0.01f, 0.01f);
        return position;
    }
    */

    GameObject FindNearestBeacon()
    {
        GameObject beacon;
        float[] distances = new float[beacons.Length];
        
        for (int i = 0; i < beacons.Length; i++) distances[i] = 
            Distance2D(transform.position, beacons[i].transform.position);

        beacon = beacons[System.Array.IndexOf(distances, distances.Min())];
        return beacon;
    }


    float Distance2D(Vector3 start, Vector3 end)
    {
        float d;
        float x = start.x - end.x;
        float z = start.z - end.z;
        d = Mathf.Sqrt(x * x + z * z);
        return d;
    }


    void Visualize()
    {
        Debug.DrawRay(transform.position, direction, Color.red);
        Debug.DrawRay(transform.position, transform.forward, Color.green);
    }
}
