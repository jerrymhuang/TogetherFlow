using System.Linq;
using UnityEngine;


public class AttentionalAgent : Agent
{

    float distractability = 0.5f;
    bool isAttentive = true;

    GameObject[] beacons;
    float[] distancesToBeacons;
    int nearestBeacon;


    void Start()
    {
        (beacons, distancesToBeacons) = InitializeBeacons();
    }


    private void Update()
    {
        if (isAttentive)
        {
            AttendTo(nearestBeacon);
        }
        else UnattendFrom(nearestBeacon, distractability);
    }

    void AttendTo(int beaconId)
    {
        /* TODO */
    }

    private void UnattendFrom(int beaconId, float distractability)
    {
        /* TODO */
    }

    (GameObject[] beacons, float[] distancesToBeacons) InitializeBeacons()
    {
        beacons = GameObject.FindGameObjectsWithTag("Beacon");
        distancesToBeacons = new float[beacons.Length];
        return (beacons, distancesToBeacons);
    }

    int FindNearestBeacon()
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
