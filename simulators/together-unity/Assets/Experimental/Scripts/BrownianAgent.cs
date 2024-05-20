using System.Collections.Generic;
using UnityEngine;


public class BrownianAgent : MonoBehaviour
{
    /* This is a MonoBehaviour script, for now. 
     * Later, this will become inherited from Agent. */
    GameObject room;
    GameObject[] targets;
    GameObject[] friends;
    float[] distances;


    float angle;
    float speed = 0.1f;
    float dA, dR; // Diffusivity constants D for the brownian model
    float detectionDistance;

    Vector3 centerOfFriends;
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
        if (friends.Length > 0)
        {
            centerOfFriends = FindCenterOfFriends();
            Debug.Log(centerOfFriends);
        }
        WalkAround();
    }


    /// <summary>
    /// Implements the active brownian particle model.
    /// </summary>
    void WalkAround()
    {
        Vector3 pos = transform.position;
        angle = Random.Range(-Mathf.PI, Mathf.PI);
        direction = new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));

        pos += speed * direction;

        if (pos.x > 4f) pos.x = 4f;
        if (pos.x < -4f) pos.x = -4f;
        if (pos.z > 5f) pos.z = 5f;
        if (pos.z < -5f) pos.z = -5f;
        
        transform.position = pos;

        transform.rotation = Quaternion.Euler(0f, Mathf.Rad2Deg * angle, 0f);

        Debug.DrawRay(transform.position, direction, Color.yellow);
    }


    /// <summary>
    /// Initialize all game objects and parameters to track.
    /// </summary>
    void Initialize()
    {
        // Find rooms and targets
        room = GameObject.FindGameObjectWithTag("Room");
        targets = GameObject.FindGameObjectsWithTag("Target");

        // Find all other agents (co-occupants of the room; or friends)
        friends = GameObject.FindGameObjectsWithTag("Agent");

        // Initialize tracking for distances to targets.
        // For now, assume that all targets in the environment are tracked.
        distances = new float[targets.Length];
    }


    /// <summary>
    /// Tracking function for the closest target for determining the drift
    /// of the agent at any given time.
    /// </summary>
    /// <returns>
    /// closestTarget   : int
    ///     index of the closest target.
    /// </returns>
    int FindClosestTarget()
    {
        int closestTarget = -1;
        float closestDistance;
        for (int i = 0; i < targets.Length; i++)
        {
            distances[i] = Vector3.Distance(transform.position, targets[i].transform.position);
        }

        closestDistance = Mathf.Min(distances);

        for (int i = 0; i < targets.Length; i++)
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
    Vector3 FindCenterOfFriends()
    {
        Vector3 center = transform.position;
        int numNearbyFriends = 0;
        for (int i = 0; i < friends.Length; i++)
        {
            float distance = Vector3.Distance(
                transform.position, friends[i].transform.position
            );
            if (distance < detectionDistance)
            {
                numNearbyFriends++;
                center += friends[i].transform.position;
            }
        }
        center = center / numNearbyFriends;
        return center;
    }


    Vector3 RelativePosition() => transform.position - room.transform.position;
}
