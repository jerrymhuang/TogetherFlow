using System.Collections.Generic;
using UnityEngine;

public class SampleAgentGroup : MonoBehaviour
{


    enum AgentHeight { Enable, Disable }
    enum GizmoType { Position, Rotation }

    [Header("Required")]
    [SerializeField]
    BrownianAgent prefab;

    /* This is an array of agents, for now, 
     * with a fixed number of agents.
     * Later, this would change to a list of agents 
     * with random initialization. */

    [SerializeField, Range(0, 49)]
    int numAgents = 12;

    [Header("Optional")]
    AgentHeight agentHeight = AgentHeight.Enable;

    [Header("Diagnosis")]
    [SerializeField]
    GizmoType gizmoType;

    BrownianAgent[] agents;

    /* Reserved for later */
    // List<SampleAgent> agents;
    // enum InitialPositionType { Random, Entry, Center }
    // InitialPositionType initialPositionType;

    List<Vector3> initialPositions;
    Vector3 initialPosition;
    Vector3 initialRotation;


    void Awake()
    {
        InitializePositions();
        agents = new BrownianAgent[numAgents];
        //agents = new List<SampleAgent>();
        for (int i = 0; i < numAgents; i++)
        {
            initialPosition = initialPositions[i];

            initialRotation = new Vector3(0f, Random.Range(-180f, 180f), 0f);

            // initialPositionType = (InitialPositionType)Random.Range(0, 2);
            BrownianAgent agent = Instantiate(prefab);

            agent.gameObject.transform.position = initialPosition;
            agent.gameObject.transform.rotation = Quaternion.Euler(initialRotation);
            agent.gameObject.transform.parent = transform;
        }

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void InitializePositions()
    {
        initialPositions = new List<Vector3>();

        while (initialPositions.Count < numAgents)
        {
            bool candidate = true;
            Vector3 initialPosition = new Vector3(
                Random.Range(-4f, 4f),
                (agentHeight == AgentHeight.Enable) ? 1.6f : 0f,
                Random.Range(-5f, 5f)
            );
            if (initialPositions.Count != 0)
            {
                for (int i = 0; i < initialPositions.Count; i++)
                {
                    if (Vector3.Distance(initialPosition, initialPositions[i]) < .5f)
                    {
                        candidate = false;
                        break;
                    }
                }
            }
            if (candidate) initialPositions.Add(initialPosition);
        }
    }
    
}
