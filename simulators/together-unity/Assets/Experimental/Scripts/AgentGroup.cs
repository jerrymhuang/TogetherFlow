using System.Collections.Generic;
using UnityEngine;

public class AgentGroup : MonoBehaviour
{

    [SerializeField]
    int numAgents = 24;

    [SerializeField]
    GameObject prefab;

    List<GameObject> agents;

    void Awake()
    {
        agents = new List<GameObject>();

        for (int i = 0; i < numAgents; i++)
        {
            GameObject agent = Instantiate(prefab);
            agent.gameObject.transform.parent = transform;
            agents.Add(agent);
        }
    }

    private void Update()
    {
        foreach (GameObject agent in agents)
        {
            if (agent.GetComponent<Agent>() != null)
            {
                agent.GetComponent<Agent>().Bound();
                //agent.GetComponent<Agent>().FlockWith(agents);
            }
        }
    }
}
