using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class IndependentAgent : MonoBehaviour
{
    // Beacons to keep track of
    GameObject room;
    GameObject[] beacons;
    float[] distancesToBeacon;

    Vector3 positionToRoom;
    Vector3 positionToBeacon;
    GameObject attendedBeacon;
    string attendedBeaconId;
    Vector3 beaconDirection;

    List<GameObject> beaconsInRange;
    float sensingDistance = 20f;
    bool isAttending;



    void Start()
    {
        room = GameObject.FindGameObjectWithTag("Room");
        beacons = GameObject.FindGameObjectsWithTag("Beacon");
        beaconsInRange = new List<GameObject>();
    }


    void FixedUpdate()
    {
        distancesToBeacon = TrackBeaconsInRange(sensingDistance);
        // distancesToBeacon = TrackBeaconDistances(beacons);

        if (distancesToBeacon.Length > 0)
        {
            attendedBeacon = FindNearestBeacon(distancesToBeacon);
            attendedBeaconId = attendedBeacon.name;
            transform.forward = Vector3.RotateTowards(
                transform.forward, Approach(attendedBeacon), 0.1f, 0f
            );
        }
        else
        {
            WalkAway();
        }
        // Debug.Log(attendedBeacon.name);
        // Debug.DrawRay(transform.position, transform.forward, Color.yellow);
    }


    


    Vector3 Approach(
        GameObject beacon,
        float baseDrift = 0.5f,
        float scale = 0.01f
    )
    {
        Vector3 dir;

        positionToRoom = transform.localPosition;
        positionToBeacon = RelativePosition2D(
            beacon.transform.position, transform.position
        );
        float distanceToBeacon = Distance2D(
            beacon.transform.position, transform.position
        );
        dir = Vector3.Normalize(positionToBeacon);
        // Debug.Log(dir);


        float drift = baseDrift * Mathf.Log(distanceToBeacon);
        float displacement = drift * Time.deltaTime;
        positionToRoom.x += displacement * dir.x +
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();
        positionToRoom.z += displacement * dir.z +
            scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();

        transform.localPosition = Bounded(positionToRoom);

        return dir;
    }


    Vector3 WalkAway()
    {
        Vector3 dir;
        float reorientationSpeed = 
            (Random.value + 0.5f) * 0.1f * (Random.value < 0.5f ? -1f : 1f);

        positionToRoom = RelativePosition2D(
            transform.position, room.transform.position
        );

        float distanceToRoom = Distance2D(
            room.transform.position, transform.position
        );
        dir = Vector3.Normalize(positionToRoom);

        positionToRoom = transform.localPosition;
        transform.forward = Vector3.RotateTowards(
            transform.forward, Approach(room), reorientationSpeed, 0f
        );

        return dir;
    }


    float[] TrackBeaconsInRange(float distance)
    {
        float[] distances = TrackBeaconDistances(beacons);
        for (int i = 0; i < beacons.Length; i++)
        {
            if (distances[i] < distance)
            {
                if (!beaconsInRange.Contains(beacons[i]))
                {
                    beaconsInRange.Add(beacons[i]);
                }

            }
            else
            {
                if (beaconsInRange.Contains(beacons[i]))
                {
                    beaconsInRange.Remove(beacons[i]);
                }
            }
        }

        return TrackBeaconDistances(beaconsInRange);
    }


    float[] TrackBeaconDistances(GameObject[] beacons)
    {
        float[] distances = new float[beacons.Length];

        for (int i = 0; i < beacons.Length; i++) distances[i] =
            Distance2D(transform.position, beacons[i].transform.position);
        return distances;
    }


    float[] TrackBeaconDistances(List<GameObject> beacons)
    {
        float[] distances = new float[beacons.Count];

        for (int i = 0; i < beacons.Count; i++) distances[i] =
            Distance2D(transform.position, beacons[i].transform.position);

        return distances;
    }


    GameObject FindNearestBeacon(float[] d)
    {
        GameObject beacon = beacons[System.Array.IndexOf(d, d.Min())];
        return beacon;
    }


    public float Distance2D(Vector3 start, Vector3 end)
    {
        float d;
        float x = start.x - end.x;
        float z = start.z - end.z;
        d = Mathf.Sqrt(x * x + z * z);

        return d;
    }

    public Vector3 RelativePosition2D(Vector3 a, Vector3 b) 
        => new Vector3(a.x - b.x, 0f, a.z - b.z);


    public Vector3 Bounded(Vector3 position)
    {
        if (position.x < -4f) position.x = -4f + Random.Range(-0.01f, 0.01f);
        if (position.x > 4f) position.x = 4f + Random.Range(-0.01f, 0.01f);
        if (position.z < -5f) position.z = -5f + Random.Range(-0.01f, 0.01f);
        if (position.z > 5f) position.z = 5f + Random.Range(-0.01f, 0.01f);
        return position;
    }


    public Vector3 Advert(Vector3 position, float buffer)
    {
        Vector3 dir = Vector3.zero;

        if (Mathf.Abs(position.x) >= 4f - buffer || 
            Mathf.Abs(position.z) <= 5f - buffer)
        {
            
        }
        return dir;
    }

}
