using System.Collections.Generic;
using System.Linq;
using UnityEngine;


public class AttentionalAgent : Agent
{

    bool isAttentive = true;

    GameObject[] beacons;
    float[] distancesToBeacons;
    int nearestBeacon;

    // private Vector3 orientation;
    // float reorientationSpeed = 0.01f;


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
            Debug.Log(beacons[nearestBeacon].name);
            transform.forward += AttendTo(nearestBeacon);
        }

        for (int i = 0; i < beacons.Length; i++)
        {
            Debug.DrawLine(transform.position, beacons[i].transform.position, Color.cyan);
        }

        Debug.DrawRay(transform.position, transform.forward, Color.red);
    }


    public override void FlockWith(List<Agent> agents)
    {
        //base.FlockWith(agentGroup);
    }


    public override Vector3 Align(List<Agent> agents, float visualDistance = 1f, bool visualize = false)
    {
        return Vector3.zero;
    }


    public override Vector3 Amass(List<Agent> agents, float motorDistance = 1f, bool visualize = false)
    {
        return Vector3.zero;
    }


    public override Vector3 Avoid(List<Agent> agents, float socislDistance = 1f, bool visualize = false)
    {
        return Vector3.zero;
    }


    private Vector3 AttendTo(int beaconId)
    {
        Vector3 beaconLocation = 
            beacons[beaconId].transform.position - transform.position;
        

        return Vector3.RotateTowards(
            transform.forward, beaconLocation, 
            0.01f, 0f
        );
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

    public override void Bound()
    {
        base.Bound();
    }

}
