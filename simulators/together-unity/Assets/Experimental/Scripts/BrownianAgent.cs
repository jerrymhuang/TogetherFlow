using System.Linq;
using UnityEngine;


public class BrownianAgent : MonoBehaviour
{
    /* This is a MonoBehaviour script, for now. 
     * Later, this will become inherited from a more general Agent class. 
     */
    GameObject room;
    GameObject[] beacons;
    GameObject[] neighbors;
    float[] distances;


    bool move;      // [TODO] Locomotive state of the agent

    float angle;
    float speed = 0.1f;
    float perceptionDistance = 2f;
    float avoidanceDistance = 0.3f;

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
        Simulate(drift: speed);
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

        // Get beacon position (as decision boundary for the Wiener process)
        int beaconId = FindNearestBeacon();
        Vector3 beaconPosition = beacons[beaconId].transform.position;

        Vector3 pull = Vector3.Normalize(beaconPosition - agentPosition);

        // Drift diffusion with pull
        if (Vector3.Distance(agentPosition, beaconPosition) > 0f)
        {
            agentPosition.x += drift * pull.x * Time.deltaTime + scaleX * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();
            agentPosition.z += drift * pull.z * Time.deltaTime + scaleZ * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();
        }

        transform.position = agentPosition;

        // Rotate towards the new beacon
        Vector3 relativePosition = beaconPosition - transform.position;
        Quaternion rotation = Quaternion.LookRotation(relativePosition, Vector3.up);
        transform.rotation = Quaternion.Lerp(transform.rotation, rotation, Time.time * speed);

        Debug.DrawRay(agentPosition, agentRotation, Color.yellow);

        Flock();
        Bound();
        Attend();
    }


    void Flock()
    {
        Align();
        Amass();
        Avoid();
    }
    
    /// <summary>
    /// Bound the agent so that it is moving within the room.
    /// </summary>
    void Bound()
    {
        Vector3 pos = transform.localPosition;

        // The boundary is set to 1 meter away from the room perimeter.
        if (pos.x > 4f) pos.x = 4f;
        if (pos.x < -4f) pos.x = -4f;
        if (pos.z > 5f) pos.z = 5f;
        if (pos.z < -5f) pos.z = -5f;

        transform.localPosition = pos;
    }


    void Attend()
    {
        // TODO: Attention to beacons fading over time.
    }


    /// <summary>
    /// Align the agent to the same heading as their neighbors.
    /// </summary>
    void Align()
    {
        var (n, c, dir) = FindNearestNeighbors();

        transform.forward = dir;
    }


    /// <summary>
    /// Make the agents gather together in smaller groups.
    /// </summary>
    void Amass()
    {
        var (n, c, dir) = FindNearestNeighbors();

        // We want the cohesion (gathering) between agents to have a
        // smaller influence than their attraction towards the target.
        transform.position = Vector3.MoveTowards(
            transform.position, c, speed * 0.1f
        );
    }


    /// <summary>
    /// Separate the agent from others to avoid collision.
    /// </summary>
    /// <param name="scale"></param>
    void Avoid(float scale = 1f)
    {
        Vector3 avoidance = Vector3.zero;

        for (int i = 0; i < neighbors.Length; i++)
        {
            float dx = transform.position.x - neighbors[i].transform.position.x;
            float dz = transform.position.z - neighbors[i].transform.position.z;

            float proximity = dx * dx + dz * dz;

            if (proximity < avoidanceDistance)
            {
                avoidance.x += dx;
                avoidance.z += dz;
            }
        }

        Debug.Log("Avoidance: " + avoidance);
        // avoidance = Vector3.Normalize(avoidance);
        transform.Translate(avoidance * scale);
    }


    int FindNearestBeacon()
    {
        for (int i = 0; i < beacons.Length; i++)
        {
            distances[i] = Vector3.Distance(
                transform.position, beacons[i].transform.position
            );
        }

        int closestTarget = distances.ToList().IndexOf(distances.Min());
        return closestTarget;
    }


    (int n, Vector3 center, Vector3 heading) FindNearestNeighbors()
    {
        int n = 0;
        Vector3 center = transform.position;
        Vector3 heading = transform.forward;

        if (neighbors.Length > 0)
        {
            for (int i = 0; i < neighbors.Length; i++)
            {
                float distance = Vector3.Distance(
                    transform.position, neighbors[i].transform.position
                );

                if (distance < perceptionDistance)
                {
                    n++;
                    center += neighbors[i].transform.position;
                    heading += neighbors[i].transform.forward;
                }
            }
        }

        center = center / n;
        heading = heading / n;

        return (n, center, heading);
    }

}
