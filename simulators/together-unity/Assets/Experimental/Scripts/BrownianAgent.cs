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
    float speed;
    float diffusivity; // A diffusivity constant D for the brownian model


    private void Awake()
    {        
        // Tag the agent
        transform.gameObject.tag = "Agent";
    }

    private void Start()
    {
        Initialize();        
    }

    void Update()
    {
        WalkAround();
    }


    /// <summary>
    /// Implements the active brownian particle model.
    /// </summary>
    void WalkAround()
    {


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
}
