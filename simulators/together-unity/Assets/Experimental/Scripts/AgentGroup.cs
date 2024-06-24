using System.Collections.Generic;
using UnityEngine;

public class AgentGroup : MonoBehaviour
{
    [SerializeField]
    bool isAgentic = true;

    [SerializeField]
    int numAgents = 6;

    [SerializeField]
    Agent agentPrefab;

    [SerializeField]
    GameObject prefab;

    List<Agent> agents;
    List<GameObject> a;

    void Awake()
    {
        if (isAgentic)
        {
            agents = new List<Agent>();

            for (int i = 0; i < numAgents; i++)
            {
                Agent agent = Instantiate(agentPrefab);
                agent.gameObject.name = "A" + i.ToString();
                agent.gameObject.transform.parent = transform;
                agents.Add(agent);
            }
        }
        else
        {
            a = new List<GameObject>();

            for (int i = 0; i < numAgents; i++)
            {
                GameObject agent = Instantiate(prefab);
                agent.gameObject.transform.parent = transform;
                a.Add(agent);
            }
        }
    }

    private void Update()
    {
        foreach (Agent agent in agents)
        {
            if (agent.GetComponent<Agent>() != null)
            {
                //agent.GetComponent<Agent>().Bound();
                agent.GetComponent<Agent>().FlockWith(agents);
            }
        }
    }
}
