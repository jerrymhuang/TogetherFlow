using UnityEngine;
using System.Linq;
using System.Collections.Generic;

public class AttentionalAgent : Agent
{
    // Attention weights
    float selfAttentionWeight = 0.5f;
    float jointAttentionWeight = 0.5f;

    // Additional properties for attentional agent
    float attentionDistance;
    float maxAttentionDistance = 30f;

    // Spatial information to keep track of
    GameObject room;
    GameObject[] beacons;
    GameObject currentBeacon, previousBeacon;
    float distanceToBeacon;
    Vector3 relativePositionToRoom;
    Vector3 relativePositionToBeacon;
    Vector3 beaconDirection;

    // Tracking variables
    bool attend = false;            // Agent state of attention onset
    bool advert = false;            // Agent state of attention switching
    float timer = 0f;
    float attentionSpan = 3f;       // Duration of individual fixation
    float switchingTime = 0.5f;     // Duration of arousal and depletion

    bool visualize = true;


    void Start()
    {
        room = GameObject.FindGameObjectWithTag("Room");
        beacons = GameObject.FindGameObjectsWithTag("Beacon");

        // Sample attention distance individually
        attentionDistance = 
            Random.Range(visualDistance * 2f, maxAttentionDistance);
    }


    private void Update()
    {
        UpdateAttentionWeights();
    }


    /// <summary>
    /// Updates the weight between joint attention and self attention.
    /// </summary>
    public virtual void UpdateAttentionWeights()
    {
        jointAttentionWeight = 1f - selfAttentionWeight;
    }


    void FixedUpdate()
    {
        UpdateVelocity();

        currentBeacon = FindNearestBeacon();
        attend = (currentBeacon != previousBeacon);

        if (attend)
        {
            distanceToBeacon = Distance2D(
                transform.position, currentBeacon.transform.position
            );
        }

        Vector3 attention = Attend(currentBeacon, attentionDistance);
        // transform.forward = attention;
        if (visualize) Visualize();
    }


    public override void FlockWith(List<Agent> agentGroup)
    {
        base.FlockWith(agentGroup);

        Vector3 attention = Attend(currentBeacon, attentionDistance);
        acceleration += attention;

    }


    /// <summary>
    /// Implements the 2D Euler-Maruyama scheme for random walk 
    /// with distance-dependent drift.
    /// </summary>
    /// <param name="beacon"></param>
    void AttendTo(GameObject beacon, float baseDrift = 0.125f, float scale = 0.1f)
    {

        relativePositionToRoom = transform.localPosition;
        relativePositionToBeacon = RelativePosition2D(
            beacon.transform.position, transform.position
        );
        beaconDirection = Vector3.Normalize(relativePositionToBeacon);
        // Debug.Log("beaconDirection: " + beaconDirection);

        float drift = baseDrift * distanceToBeacon;
        float displacement = drift * Time.deltaTime;
        relativePositionToRoom.x += displacement * beaconDirection.x +
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();
        relativePositionToRoom.z += displacement * beaconDirection.z +
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();


        transform.localPosition = Bounded(relativePositionToRoom);
        transform.forward = Vector3.RotateTowards(
            transform.forward, relativePositionToBeacon, 0.05f, 0f
        );

    }


    Vector3 Attend(
        GameObject beacon,
        float baseDrift = 0.125f,
        float scale = 0.1f,
        float rotationSpeed = 0.01f
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

        relativePositionToRoom = transform.localPosition;
        relativePositionToBeacon = RelativePosition2D(
            beacon.transform.position, transform.position
        );
        dir = Vector3.Normalize(relativePositionToBeacon);

        float drift = baseDrift * distanceToBeacon;
        float displacement = drift * Time.deltaTime;
        relativePositionToRoom.x += displacement * beaconDirection.x +
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();
        relativePositionToRoom.z += displacement * beaconDirection.z +
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();

        transform.localPosition = Bounded(relativePositionToRoom);
        
        return selfAttentionWeight * dir;
    }


    Vector3 Advert(float rotationSpeed = 0.01f)
    {
        Vector3 dir = Vector3.RotateTowards(
            transform.forward, relativePositionToBeacon, rotationSpeed, 0f
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
        //Debug.DrawRay(transform.position, beaconDirection, Color.green);
        //Debug.DrawRay(transform.position, alignment, Color.magenta);
        //Debug.DrawRay(transform.position, cohesion, Color.cyan);
        //Debug.DrawRay(transform.position, separation, Color.yellow);

        Debug.Log(
            "beaconDirection: " + beaconDirection + " | " +
            "alignment: " + alignment + " | " +
            "cohesion: " + cohesion + " | " +
            "separation: " + separation
        );
        //Debug.DrawRay(transform.position, transform.forward, Color.green);
    }


    public Vector3 RelativePosition2D(Vector3 a, Vector3 b)
    => new Vector3(a.x - b.x, 0f, a.z - b.z);

    public override Vector3 Bounded(Vector3 position)
    {
        if (position.x < -4f) position.x = -4f + Random.Range(-0.01f, 0.01f);
        if (position.x > 4f) position.x = 4f + Random.Range(-0.01f, 0.01f);
        if (position.z < -5f) position.z = -5f + Random.Range(-0.01f, 0.01f);
        if (position.z > 5f) position.z = 5f + Random.Range(-0.01f, 0.01f);
        return position;
    }

}
