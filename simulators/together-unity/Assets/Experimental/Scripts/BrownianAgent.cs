using System;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;


public class BrownianAgent : MonoBehaviour
{
    /* This is a MonoBehaviour script, for now. 
     * Later, this will become inherited from Agent. */
    GameObject room;
    GameObject[] beacons;
    GameObject[] neighbors;
    float[] distances;

    bool move;
    float angle;
    float speed = 0.1f;
    float detectionDistance = 2f;
    float avoidanceDistance = 1f;

    Vector3 centerOfNeighbors;
    Vector3 direction;


    private void Awake()
    {        
        // Tag the agent
        transform.gameObject.tag = "Agent";
    }

    private void Start()
    {
        Initialize();        
    }

    void FixedUpdate()
    {
        /*
        if (neighbors.Length > 0)
        {
            //Debug.Log(transform.position + " " + neighbors[0].transform.position);
            centerOfNeighbors = CenterOfNeighbors();
            // Debug.Log(centerOfNeighbors);
        }
        */
        Simulate();
    }


    /// <summary>
    /// Initialize all game objects and parameters to track.
    /// </summary>
    void Initialize()
    {
        // Find rooms and targets
        room = GameObject.FindGameObjectWithTag("Room");
        beacons = GameObject.FindGameObjectsWithTag("Beacon");

        // Find all other agents (co-occupants of the room; or Neighbors)
        neighbors = GameObject.FindGameObjectsWithTag("Agent");

        // Initialize tracking for distances to targets.
        // For now, assume that all targets in the environment are tracked.
        distances = new float[beacons.Length];
    }


    void Simulate(float drift = 1f, float scaleX = 0.1f, float scaleZ = 0.1f)
    {   
        // Get agent position and rotation
        Vector3 agentPosition = transform.position;
        Vector3 agentRotation = transform.forward;
        Debug.Log(agentRotation);

        // Get beacon position (as decision boundary for the Wiener process)
        int beaconId = FindNearestBeacon();
        Vector3 beaconPosition = beacons[beaconId].transform.position;

        Vector3 pull = Vector3.Normalize(beaconPosition - agentPosition);

        // Drift diffusion with pull
        if (Vector3.Distance(agentPosition, beaconPosition) > 0f)
        {
            agentPosition.x += drift * pull.x * Time.deltaTime + scaleX * Mathf.Sqrt(Time.deltaTime) * GaussianRNG.Sample();
            agentPosition.z += drift * pull.z * Time.deltaTime + scaleZ * Mathf.Sqrt(Time.deltaTime) * GaussianRNG.Sample();
        }

        transform.position = agentPosition;
        transform.LookAt(beacons[beaconId].transform.position);
        Debug.DrawRay(agentPosition, agentRotation, Color.yellow);
        BoundPosition();
        Separate(0.5f);
        Align();
    }


    void BoundPosition()
    {
        Vector3 pos = transform.localPosition;

        if (pos.x > 4f) pos.x = 4f;
        if (pos.x < -4f) pos.x = -4f;
        if (pos.z > 5f) pos.z = 5f;
        if (pos.z < -5f) pos.z = -5f;

        transform.localPosition = pos;
    }

    void Separate(float scale = 1f)
    {
        Vector3 avoidance = Vector3.zero;

        for (int i = 0; i < neighbors.Length; i++)
        {
            float proximity = Vector3.Distance(
                transform.position, neighbors[i].transform.position
            );
            if (proximity < 0.3f)
            {
                avoidance += transform.position - neighbors[i].transform.position;
            }
        }

        avoidance = Vector3.Normalize(avoidance);
        transform.Translate(avoidance);
    }

    void Align()
    {
        // TODO: Implement alignment
    }

    /// <summary>
    /// Tracking function for the closest target for determining the drift
    /// of the agent at any given time.
    /// </summary>
    /// <returns>
    /// closestTarget   : int
    ///     index of the closest target.
    /// </returns>
    int FindNearestBeacon()
    {
        int closestTarget = -1;
        float closestDistance;
        for (int i = 0; i < beacons.Length; i++)
        {
            distances[i] = Vector3.Distance(transform.position, beacons[i].transform.position);
        }

        closestDistance = Mathf.Min(distances);

        for (int i = 0; i < beacons.Length; i++)
        {
            if (distances[i] == closestDistance)
            {
                closestTarget = i;
                break;
            }
        }
        return closestTarget;
    }


    /// <summary>
    /// Find the center of all surrounding agents within a distance constraint.
    /// </summary>
    /// <returns>
    /// center      : Vector3
    ///     Center position of all surrounding agents within a distance.
    /// </returns>

    (int n, Vector3 center, float direction) Flocks()
    {
        int n = 0;
        Vector3 center = transform.position;
        float direction = transform.rotation.y;

        for (int i = 0; i < neighbors.Length; i++)
        {
            float distance = Vector3.Distance(
                transform.position, neighbors[i].transform.position
            );

            if (distance < detectionDistance)
            {
                n++;
                center += neighbors[i].transform.position;
                direction += neighbors[i].transform.rotation.y;
            }
        }

        center = center / n;
        direction = direction / n;

        return (n, center, direction);
    }


    Vector3 PositionInRoom() => transform.position - room.transform.position;
}
