using System.Collections.Generic;
using UnityEngine;

public class AgentGroup : MonoBehaviour
{

    [SerializeField]
    int numAgents = 6;

    [SerializeField]
    Agent prefab;

    List<Agent> agents;

    void Awake()
    {
        agents = new List<Agent>();

        for (int i = 0; i < numAgents; i++)
        {
            Agent agent = Instantiate(prefab);
            agent.gameObject.transform.parent = transform;
            agents.Add(agent);
        }
    }

    private void Update()
    {
        foreach (Agent agent in agents)
        {
            if (agent.GetComponent<Agent>() != null)
            {
                agent.GetComponent<Agent>().Bound();
                //agent.GetComponent<Agent>().FlockWith(agents);
            }
        }
    }
}
