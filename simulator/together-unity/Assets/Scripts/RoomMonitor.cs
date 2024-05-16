using UnityEngine;


public class RoomMonitor : MonoBehaviour
{
    [SerializeField]
    GameObject controller;

    [SerializeField]
    bool diagnose = false;

    GameObject[] targets;

    public float ControllerHeading { get; private set; }
    public float DistanceToTarget { get; private set; }
    public float[] DistancesToTargets { get; private set; }
    public float[] RelativeAngles { get; private set; }
    public Vector3[] RelativePositions { get; private set; }


    int targetCount;


    private void OnEnable()
    {
        targets = GameObject.FindGameObjectsWithTag("Target");
        targetCount = targets.Length;
    }


    // Update is called once per frame
    void Update()
    {
        ControllerHeading = controller.transform.rotation.eulerAngles.y;
        GetSpatialData(controller, targets);
    }

    void GetSpatialData(GameObject controller, GameObject[] targets)
    {
        DistancesToTargets = new float[targetCount];
        RelativeAngles = new float[targetCount];
        RelativePositions = new Vector3[targetCount];

        Vector3 referencePosition;
        float referenceAngle;

        for (int i = 0; i < targetCount; i++)
        {
            /* Calculate a reference position between the controller and the sound source
                * without taking into account the controller's heading. */
            referencePosition = new Vector3(
                targets[i].transform.position.x - controller.transform.position.x, 0f,
                targets[i].transform.position.z - controller.transform.position.z);

            /* Determine the absolute angle of the virtual sound source based upon the 
                * positive x-axis of the world coordinates (right-hand rule). */
            referenceAngle = Vector3.SignedAngle(
                referencePosition, Vector3.right, Vector3.up
            );

            /* Calculate the distance between the sound source and the controller. */
            DistancesToTargets[i] = Vector3.Distance(
                targets[i].transform.position, controller.transform.position
            );

            /* Calculate the relative angle between the controller's heading 
                * and the direction of the virtual sound source. */
            RelativeAngles[i] = Mathf.Deg2Rad * (ControllerHeading + referenceAngle);

            /* Finally, calculate the relative positions based upon the 
                * relative angle. */
            RelativePositions[i] = new Vector3(
                DistancesToTargets[i] * Mathf.Cos(RelativeAngles[i]), 0f,
                DistancesToTargets[i] * Mathf.Sin(RelativeAngles[i])
                );

            if (diagnose)
            {
                if (i == 0) Debug.Log(targets[i].name + " | " + RelativePositions[i]);
            }
        }
    }
}