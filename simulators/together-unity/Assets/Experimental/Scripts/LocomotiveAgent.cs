using UnityEngine;
using System.Linq;
using System.Net;

public class LocomotiveAgent : Agent
{

    GameObject[] beacons;
    float minAttentionDistance;
    float maxAttentionDistance;
    float distanceToBeacon;

    GameObject beacon;
    Vector3 positionToRoom;
    Vector3 positionToBeacon;
    Vector3 direction;

    bool visualize = true;
    float timer = 0f;
    float maxAttentionTime = 3f;


    void Start()
    {
        transform.localPosition = Vector3.right * Random.Range(-4f, 4f) + 
                                  Vector3.forward * Random.Range(-5f, 5f);
        beacons = GameObject.FindGameObjectsWithTag("Beacon");

        // Sample attention distance individually
        minAttentionDistance = Random.Range(5f, 10f);
        maxAttentionDistance = Random.Range(minAttentionDistance + 20f, minAttentionDistance + 40f);
        
    }


    void FixedUpdate()
    {
        UpdateVelocity();

        beacon = FindNearestBeacon();
        distanceToBeacon = Distance2D(transform.position, beacon.transform.position);

        if (distanceToBeacon < maxAttentionDistance &&
            distanceToBeacon > minAttentionDistance)
        {
            // Debug.Log("Attending");

            AttendTo(beacon);
            timer += Time.deltaTime;
            Debug.Log(timer);
            
            if (timer > maxAttentionTime)
            {
                Debug.Log("Losing Attention");
                Unattend();
            }
        }
        else
        {
            // Debug.Log("Unattending");
        }

        if (visualize) Visualize();
    }


    /// <summary>
    /// Implements the 2D Euler-Maruyama scheme for random walk with distance-dependent drift.
    /// </summary>
    /// <param name="beacon"></param>
    void AttendTo(GameObject beacon, float baseDrift = 0.125f, float scale = 0.1f)
    {

        positionToRoom = transform.localPosition;
        positionToBeacon = beacon.transform.position - transform.position;
        direction = Vector3.Normalize(positionToBeacon);
        // Debug.Log("Direction: " + direction);

        float drift = baseDrift * distanceToBeacon;
        float displacement = drift * Time.deltaTime;
        positionToRoom.x += displacement * direction.x + 
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();
        positionToRoom.z += displacement * direction.z + 
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();


        transform.localPosition = Bounded(positionToRoom);
        transform.forward = Vector3.RotateTowards(
            transform.forward, positionToBeacon, 0.05f, 0f
        );
    }




    /// <summary>
    /// Reorient to a random location within the room.
    /// </summary>
    /// <param name="beacon"></param>
    void Unattend()
    {
        float drift = RNG.Gaussian();
        float scale = 0.05f;
        transform.Rotate(Vector3.up * (drift * Time.deltaTime + scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian()));
        /*
        Vector3 direction = 
            Vector3.forward * Random.Range(-5f, 5f) + 
            Vector3.right * Random.Range(-4f, 4f);
        transform.forward = Vector3.RotateTowards(transform.forward, direction, 0.05f, 0f);
        */
    }


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
        //Debug.DrawRay(transform.position, transform.forward, Color.green);
    }


    public override Vector3 Bounded(Vector3 position)
    {
        if (position.x < -4f) position.x = -4f + Random.Range(-0.01f, 0.01f);
        if (position.x > 4f) position.x = 4f + Random.Range(-0.01f, 0.01f);
        if (position.z < -5f) position.z = -5f + Random.Range(-0.01f, 0.01f);
        if (position.z > 5f) position.z = 5f + Random.Range(-0.01f, 0.01f);
        return position;
    }


    public override void Bound()
    {
        
    }
}
