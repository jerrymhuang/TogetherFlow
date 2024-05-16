using UnityEngine;


public class GenericAgent : MonoBehaviour
{
    bool diagnose = true;
    Transform parent;

    public GameObject[] targets { get; private set; }
    public int currentTargetId { get; private set; }
    public float distanceToTarget { get; private set; }
    public float[] distancesToTargets { get; private set; }
    public Vector3 lookDirection { get; private set; }

    Quaternion orientation;

    int previousTargetId;
    int currentTurnSteps = 0, totalTurnSteps = 30;
    float currentTurnRatio;

    GameObject room;
    Vector3 roomPosition;


    void OnEnable()
    {
        room = GameObject.FindGameObjectWithTag("Room");

        transform.localRotation = Random.rotation;
        if (transform.parent != null) parent = transform.parent;
        targets = GameObject.FindGameObjectsWithTag("Target");
        distancesToTargets = new float[targets.Length];
    }


    void FixedUpdate()
    {
        roomPosition = room.transform.position;
        currentTargetId = NearestTargetId();


        if (Vector3.Distance(roomPosition, transform.position) < 4f)
        {
            Debug.Log("Locomotive target: " + currentTargetId);
            transform.position = Vector3.MoveTowards(
                transform.position, targets[currentTargetId].transform.position, 0.01f
            );
        }


        if (currentTargetId != previousTargetId)
        {
            // Determine how many degrees to turn
            currentTurnRatio = (float)currentTurnSteps / totalTurnSteps;

            // Make the turn
            lookDirection = Vector3.LerpUnclamped(
                NoisyLookDirection(targets[previousTargetId], 1f),
                NoisyLookDirection(targets[currentTargetId], 1f),
                currentTurnRatio);
            orientation = Quaternion.LookRotation(lookDirection, Vector3.up);
            transform.localRotation = orientation;

            // Increment the degrees to turn
            currentTurnSteps = (currentTurnSteps + 1) % totalTurnSteps;

            // Update the target
            if (currentTurnSteps == 0f) 
                previousTargetId = currentTargetId;
        }
        else
        {
            lookDirection = NoisyLookDirection(targets[currentTargetId], 3f);
            orientation = Quaternion.LookRotation(lookDirection, Vector3.up);
            transform.localRotation = orientation;
        }

        Debug.Log("Global target: " + currentTargetId);

        if (diagnose)
        {
            if (parent != null) Debug.Log(lookDirection);
            VisualizeSpatialBehaviors();
        }
    }


    void VisualizeSpatialBehaviors()
    {
        Debug.DrawRay(transform.position, lookDirection, Color.magenta);
    }


    int NearestTargetId()
    {
        float distance = Mathf.Infinity;
        int nearestId = -1;
        for (int i = 0; i < targets.Length; i++)
        {
            float d = Vector3.Distance(
                transform.position, targets[i].transform.position);
            distancesToTargets[i] = d;

            if (d < distance)
            {
                distance = d;
                nearestId = i;
            }
        }
        distanceToTarget = distance;
        return nearestId;
    }


    Vector3 NoisyLookDirection(GameObject target, float noisiness)
    {
        float noise = Random.value * noisiness;
        Vector3 noiseVector = 
            Vector3.right * Mathf.Cos(noise) + Vector3.forward * Mathf.Sin(noise);
        return (target.transform.position - transform.position + noiseVector).normalized;
    }
}