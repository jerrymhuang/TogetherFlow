using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;


public class AttentionalAgent : Agent
{

    float distractability = 0.5f;
    bool isAttentive = true;

    GameObject[] beacons;
    float[] distancesToBeacons;
    int nearestBeacon;

    Vector3 orientation;
    float reorientationSpeed;


    void Start()
    {
        var (beacons, distancesToBeacons) = InitializeBeacons();
        Debug.Log(beacons.Length);
    }


    private void Update()
    {
        if (isAttentive)
        {
            nearestBeacon = FindNearestBeacon();
            AttendTo(nearestBeacon);
        }
        else UnattendFrom(nearestBeacon, distractability);
    }

    public override void FlockWith(List<Agent> agentGroup)
    {
        //base.FlockWith(agentGroup);
    }

    void AttendTo(int beaconId)
    {
        /* TODO */
        Vector3 beaconLocation = 
            beacons[beaconId].transform.position - transform.position;

        

        orientation = Vector3.RotateTowards(
            transform.forward, beaconLocation, reorientationSpeed * Time.deltaTime, 0f
        );

        Debug.DrawRay(transform.position, orientation, Color.red);

        transform.localRotation = Quaternion.LookRotation( orientation );
    
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
