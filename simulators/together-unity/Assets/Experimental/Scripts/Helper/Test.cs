using System.Numerics;
using UnityEngine;
using Quaternion = UnityEngine.Quaternion;
using Vector3 = UnityEngine.Vector3;

public class Test : MonoBehaviour
{
    private GameObject[] beacons;
    private int beaconId;

    void Start()
    {
        transform.position = new Vector3(
            Random.Range(-10f, 10f), 0f, Random.Range(-10f, 10f));

        transform.rotation = Quaternion.Euler(
            new Vector3(0f, Random.Range(-180f, 180f), 0f)
        );

        beacons = GameObject.FindGameObjectsWithTag("Beacon");
        beaconId = Random.Range(0, beacons.Length);
    }

    void Update()
    {
        LookTowards(beacons[beaconId]);
        Debug.Log(beacons[beaconId].name);
        Debug.DrawRay(transform.position, transform.forward);
    }

    void LookTowards(GameObject beacon)
    {
        Vector3 relativePosition = beacon.transform.position - transform.position;
        transform.forward = Vector3.RotateTowards(transform.forward, relativePosition, 0.01f, 0.01f);
    }
}
