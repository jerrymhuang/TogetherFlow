using System.Collections.Generic;
using UnityEngine;

public class AgentGroup : MonoBehaviour
{

    [SerializeField]
    int numAgents = 24;

    [SerializeField]
    FlockingAgent prefab;

    List<FlockingAgent> agents;

    void Awake()
    {
        agents = new List<FlockingAgent>();

        for (int i = 0; i < numAgents; i++)
        {
            FlockingAgent agent = Instantiate(prefab);
            agent.gameObject.transform.parent = transform;
            agents.Add(agent);
        }
    }

    private void Update()
    {
        foreach (FlockingAgent agent in agents)
        {
            agent.Bound();
            agent.FlockWith(agents);
        }
    }
}
