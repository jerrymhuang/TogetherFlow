using UnityEngine;
using System.IO;
using System.Collections;
using System;


public class SpatialDataCollector : MonoBehaviour
{
    [SerializeField]
    GenericAgentGroup agentGroup;

    enum Focus { Targets, Agents, Hybrid }

    [SerializeField]
    Focus focus;

    enum DatasetType { Training, Validation, Testing, Development }

    [SerializeField]
    DatasetType datasetType;

    string filePath = "";
    bool write = false;
    bool header = false;

    GameObject room;
    GameObject[] agents;
    GameObject[] targets;

    RoomMonitor roomMonitor;

    private void Start()
    {
        string fileName = "/Data/" + 
            Enum.GetName(typeof(Focus), focus).ToLower() + "_" + 
            Enum.GetName(typeof(DatasetType), datasetType).ToLower() + "_" + 
            "data.csv";

        filePath = Application.dataPath + fileName;

        if (!File.Exists(filePath))
        {
            File.Create(filePath);
        }
        Debug.Log(filePath);

        room = GameObject.FindGameObjectWithTag("Room");
        agents = GameObject.FindGameObjectsWithTag("Agent");
        targets = GameObject.FindGameObjectsWithTag("Target");
        Debug.Log(room.name + " " + agents.Length + targets.Length);

        roomMonitor = transform.gameObject.GetComponent<RoomMonitor>();
    }


    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C)) write = !write;
        if (write)
        {
            if (header == false)
            {
                WriteHeader();
                header = true;
            }
            StartCoroutine(WriteData());
        }
    }


    public void WriteHeader()
    {
        TextWriter writer = new StreamWriter(filePath, false);

        string header = "timeStamp;roomX;roomZ;roomAngle;";

        if (focus == Focus.Agents)
            header += "agentId;agentX;agentZ;agentLookX;agentLookZ;targetId;targetName;distanceToTarget";
        else if (focus == Focus.Targets)
            header += "targetId;targetName;targetX;targetZ;targetAngle;distanceToTarget;targetAttentionCount";
        else if (focus == Focus.Hybrid)
        {
            string agentPredictors = 
                "agentId;agentX;agentZ;agentLookX;agentLookZ";
            string targetPredictors = 
                "targetId;targetName;targetX;targetZ;targetAngle;" + 
                "distanceToRoom;distanceToAgent;LookedAt;targetAttentionCount";
            string predictors = agentPredictors + ";" + targetPredictors;
            header += predictors;
        }

        writer.WriteLine(header);
        writer.Close();
    }


    public IEnumerator WriteData()
    {
        if (focus == Focus.Agents)
            WriteAgentBasedData();
        else if (focus == Focus.Targets)
            WriteTargetBasedData();
        else if (focus == Focus.Hybrid)
            WriteHybridData();

        yield return new WaitForSeconds(0.02f);
    }


    void WriteAgentBasedData()
    {
        for (int i = 0; i < agents.Length; i++)
        {
            TextWriter writer = new StreamWriter(filePath, true);
            int targetId = agents[i].GetComponent<GenericAgent>().currentTargetId;
            float distanceToTarget =
                agents[i].GetComponent<GenericAgent>().distanceToTarget;

            string data = Time.time + ";" +
                    room.transform.position.x + ";" +
                    room.transform.position.z + ";" +
                    room.transform.rotation.eulerAngles.y * Mathf.Deg2Rad + ";" + i + ";" +
                    agents[i].transform.localPosition.x + ";" +
                    agents[i].transform.localPosition.z + ";" +
                    agents[i].GetComponent<GenericAgent>().lookDirection.x + ";" +
                    agents[i].GetComponent<GenericAgent>().lookDirection.z + ";" +
                    targetId + ";" + targets[targetId].name + ";" + distanceToTarget;

            writer.WriteLine(data);
            writer.Close();
        }
    }


    void WriteTargetBasedData()
    {
        for (int i = 0; i < targets.Length; i++)
        {
            string data = Time.time + ";" +
                    room.transform.position.x + ";" +
                    room.transform.position.z + ";" +
                    room.transform.rotation.eulerAngles.y * Mathf.Deg2Rad + ";" + i + ";" +
                    targets[i].name + ";" +
                    roomMonitor.RelativePositions[i].x + ";" +
                    roomMonitor.RelativePositions[i].z + ";" +
                    roomMonitor.RelativeAngles[i] + ";" +
                    roomMonitor.DistancesToTargets[i] + ";" +
                    agentGroup.targetAttentionCounts[i];
            TextWriter writer = new StreamWriter(filePath, true);
            writer.WriteLine(data);
            writer.Close();
        }
    }


    void WriteHybridData()
    {
        for (int i = 0; i < agents.Length; i++)
        {
            for (int j = 0; j < targets.Length; j++)
            {
                string roomData = Time.time + ";" +
                    room.transform.position.x + ";" +
                    room.transform.position.z + ";" +
                    room.transform.rotation.eulerAngles.y * Mathf.Deg2Rad;

                string agentData = i + ";" +
                    agents[i].transform.localPosition.x + ";" +
                    agents[i].transform.localPosition.z + ";" +
                    agents[i].GetComponent<GenericAgent>().lookDirection.x + ";" +
                    agents[i].GetComponent<GenericAgent>().lookDirection.z;

                string targetData = j + ";" +
                    targets[j].name + ";" +
                    roomMonitor.RelativePositions[j].x + ";" +
                    roomMonitor.RelativePositions[j].z + ";" +
                    roomMonitor.RelativeAngles[j] + ";" +
                    roomMonitor.DistancesToTargets[j] + ";" + 
                    agents[i].GetComponent<GenericAgent>().distancesToTargets[j] + ";" + 
                    ((agents[i].GetComponent<GenericAgent>().currentTargetId == j) ? 1 : 0) + ";" + 
                    agentGroup.targetAttentionCounts[j];

                string data = roomData + ";" + agentData + ";" + targetData;
                    
                TextWriter writer = new StreamWriter(filePath, true);
                writer.WriteLine(data);
                writer.Close();
            }
        }
    }
}