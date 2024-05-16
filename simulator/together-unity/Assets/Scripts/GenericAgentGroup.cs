using System.Collections.Generic;
using UnityEngine;


public class GenericAgentGroup : MonoBehaviour
{
    [SerializeField]
    bool debug;

    [SerializeField]
    int radialDistance = 2;

    [SerializeField]
    Vector2 simulationRegionSize = Vector3.one;

    [SerializeField]
    int sampleRejectionThreshold = 30;

    [SerializeField]
    GameObject agentPrefab;

    public int[] targetAttentionCounts { get; private set; }

    List<Vector2> agentLocalPositions;
    List<GameObject> agents;
    GameObject[] targets;
    int[] targetIds;

    private void Awake()
    {
        agents = new List<GameObject>();
        agentLocalPositions = PoissonDiscSampler.GenerateSamples(
            radialDistance, simulationRegionSize,
            sampleRejectionThreshold
            );

        foreach (Vector2 p in agentLocalPositions)
        {
            GameObject agent = Instantiate(agentPrefab);
            agent.transform.SetParent(transform, false);
            agent.transform.localPosition = new Vector3(
                p.x - simulationRegionSize.x / 2f, 1f,
                p.y - simulationRegionSize.y / 2f);
            agent.transform.localRotation = Random.rotation;

            agents.Add(agent);
        }

        targets = GameObject.FindGameObjectsWithTag("Target");

        targetIds = new int[agents.Count];
    }


    void Start()
    {
        if (debug) InvokeRepeating("GetDiagnostics", 0f, 1f);
    }


    void FixedUpdate()
    {
        GetTargetIds();
        targetAttentionCounts = GetTargetAttentionCounts();
        if (debug) LogTargetAttentionCounts(targetAttentionCounts);
    }


    void GetDiagnostics()
    {
        for (int i = 0; i < agents.Count; i++)
        {
            int id = agents[i].GetComponent<GenericAgent>().currentTargetId;
            //Debug.Log(i.ToString() + " | " + agents[i].GetComponent<GenericAgent>().targets[id].name);
        }
    }

    void GetTargetIds()
    {
        for (int i = 0; i < agents.Count; i++)
        {
            targetIds[i] = agents[i].GetComponent<GenericAgent>().currentTargetId;
        }
    }

    int[] GetTargetAttentionCounts()
    {
        targetAttentionCounts = new int[targets.Length];
        for (int i = 0; i < targetIds.Length; i++)
        {
            int targetId = targetIds[i];
            targetAttentionCounts[targetId] += 1;
        }
        return targetAttentionCounts;
    }

    void LogTargetAttentionCounts(int[] targetAttentionCounts)
    {
        string counts = "";
        for (int i = 0; i < targetAttentionCounts.Length; i++)
        {
            counts += targetAttentionCounts[i].ToString();
            if (i != targetAttentionCounts.Length - 1) counts += " ";
        }
        Debug.Log(counts);
    }
}