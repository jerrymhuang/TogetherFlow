using UnityEngine;
using System.Linq;

public class ReorientingAgent : MonoBehaviour
{
    private GameObject[] beacons;
    private float[] distancesToBeacons;

    private int beaconId;

    void Start()
    {
        InitializeBeacons();
    }


    // Update is called once per frame
    void Update()
    {
        beaconId = FindNearestBeacon();
        Debug.Log(beaconId);
    }


    (GameObject[] beacons, float[] distancesToBeacons) InitializeBeacons()
    {
        beacons = GameObject.FindGameObjectsWithTag("Beacon");
        distancesToBeacons = new float[beacons.Length];
        return (beacons, distancesToBeacons);
    }


    private int FindNearestBeacon()
    {
        for (int i = 0; i < beacons.Length; i++)
        {
            distancesToBeacons[i] = Vector3.Distance(
                transform.position, beacons[i].transform.position
            );
        }
        int nearestBeacon =
            distancesToBeacons.ToList().IndexOf(distancesToBeacons.Min());
        return nearestBeacon;
    }
}
