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
    float diffusivity; // A diffusivity constant D for the brownian model
    float displacement;

    Vector3 centerOfFriends;


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
        pos += speed * new Vector3(
            Mathf.Cos(angle), 0f, Mathf.Sin(angle)
        );

        if (pos.x > 4f) pos.x = 4f;
        if (pos.x < -4f) pos.x = -4f;
        if (pos.z > 5f) pos.z = 5f;
        if (pos.z < -5f) pos.z = -5f;

        transform.position = pos;

        transform.rotation = Quaternion.Euler(0f, Mathf.Rad2Deg * angle, 0f);
    }

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

    Vector3 FindCenterOfFriends()
    {
        Vector3 center = Vector3.zero;
        for (int i = 0; i < friends.Length; i++)
        {
            center += friends[i].transform.position;
        }
        center = center / friends.Length;
        return center;
    }
}
