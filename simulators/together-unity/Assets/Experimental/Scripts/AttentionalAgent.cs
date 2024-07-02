using UnityEngine;
using System.Linq;
using System.Collections.Generic;

public class AttentionalAgent : Agent
{
    // Beacons to keep track of
    GameObject room;
    GameObject[] beacons;
    float distanceToBeacon;

    // Additional properties for attentional agent
    float attentionDistance;
    float maxAttentionDistance;

    Vector3 positionToRoom;
    Vector3 positionToBeacon;
    GameObject attendedBeacon;
    Vector3 direction;

    bool visualize = true;
    float timer = 0f;
    float maxAttentionTime = 3f;


    void Start()
    {
        room = GameObject.FindGameObjectWithTag("Room");
        beacons = GameObject.FindGameObjectsWithTag("Beacon");

        // Sample attention distance individually
        attentionDistance = Random.Range(visualDistance, maxAttentionDistance);
    }


    void FixedUpdate()
    {
        // UpdateVelocity();

        attendedBeacon = FindNearestBeacon();
        distanceToBeacon = 
            Distance2D(transform.position, attendedBeacon.transform.position);

        if (distanceToBeacon < maxAttentionDistance)
        {
            Debug.Log("Attending");

            // attention = Attend(attendedBeacon, attentionDistance);
            // acceleration += attention;

            timer += Time.deltaTime;
            Debug.Log(timer);
        }
        if (timer > maxAttentionTime)
        {
            Debug.Log("Losing Attention");
            Unattend();
        }


        Vector3 attention = Attend(attendedBeacon, attentionDistance);
        transform.forward = attention;
        if (visualize) Visualize();
    }


    public override void FlockWith(List<Agent> agentGroup)
    {
        base.FlockWith(agentGroup);

        Vector3 attention = Attend(attendedBeacon, attentionDistance);
        acceleration += attention;

    }


    /// <summary>
    /// Implements the 2D Euler-Maruyama scheme for random walk 
    /// with distance-dependent drift.
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


    Vector3 Attend(
        GameObject beacon,
        float baseDrift = 0.125f,
        float scale = 0.1f,
        float rotationSpeed = 1f
    )
    {

        Vector3 dir = Approach(beacon, baseDrift, scale);
        transform.forward = Advert(rotationSpeed);

        return dir;

    }


    Vector3 Approach(
        GameObject beacon, 
        float baseDrift = 0.125f, 
        float scale = 0.1f
    )
    {
        Vector3 dir;

        positionToRoom = transform.localPosition;
        positionToBeacon = beacon.transform.position - transform.position;
        dir = Vector3.Normalize(positionToBeacon);

        float drift = baseDrift * distanceToBeacon;
        float displacement = drift * Time.deltaTime;
        positionToRoom.x += displacement * direction.x +
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();
        positionToRoom.z += displacement * direction.z +
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();

        transform.localPosition = Bounded(positionToRoom);
        
        return selfAttentionWeight * dir;
    }


    Vector3 Advert(float rotationSpeed = 1f)
    {
        Vector3 dir = Vector3.RotateTowards(
            transform.forward, positionToBeacon, rotationSpeed, 0f
        );

        return dir;
    }



    /// <summary>
    /// Reorient to a random location within the room.
    /// </summary>
    /// <param name="beacon"></param>
    void Unattend()
    {
        float drift = RNG.Gaussian();
        float scale = 0.05f;
        transform.Rotate(
            Vector3.up * (drift * Time.deltaTime + 
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian()
            )
        );
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


    void Visualize()
    {
        Debug.DrawRay(transform.position, transform.forward, Color.red);
        //Debug.DrawRay(transform.position, direction, Color.green);
        //Debug.DrawRay(transform.position, alignment, Color.magenta);
        //Debug.DrawRay(transform.position, cohesion, Color.cyan);
        //Debug.DrawRay(transform.position, separation, Color.yellow);

        Debug.Log(
            "direction: " + direction + " | " +
            "alignment: " + alignment + " | " +
            "cohesion: " + cohesion + " | " +
            "separation: " + separation
        );
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

}
