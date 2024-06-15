using UnityEngine;

public class LocomotiveAgent : Agent
{

    GameObject beacon;
    float attentionDistance = 50f;
    float distanceToBeacon;

    void Start()
    {
        transform.localPosition = Vector3.right * Random.Range(-4f, 4f) + 
                                  Vector3.forward * Random.Range(-5f, 5f);
        Debug.Log(transform.localPosition);
        beacon = GameObject.FindGameObjectWithTag("Beacon");
        Debug.Log(beacon.name);
    }


    void FixedUpdate()
    {
        distanceToBeacon = Distance2D(transform.position, beacon.transform.position);
        if (distanceToBeacon < attentionDistance) AttendTo(beacon);
        else UnattendFrom(beacon);
    }


    /// <summary>
    /// Implements the 2D Euler-Maruyama scheme for random walk with distance-dependent drift.
    /// </summary>
    /// <param name="beacon"></param>
    void AttendTo(GameObject beacon, float baseDrift = 0.5f, float decay = 0.01f, float scale = 0.1f)
    {

        Vector3 position = transform.localPosition;
        Vector3 direction = Vector3.Normalize(beacon.transform.position - transform.position);
        float distanceToBeacon = Distance2D(transform.position, beacon.transform.position);
        float drift = baseDrift * distanceToBeacon * decay;
        float displacement = drift * Time.deltaTime;
        position.x += displacement * direction.x + scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();
        position.z += displacement * direction.z + scale * Mathf.Sqrt(Time.deltaTime) * RNG.Gaussian();

        Debug.DrawRay(transform.position, direction, Color.red);

        transform.localPosition = Bound(position);
        transform.forward = Vector3.RotateTowards(transform.forward, beacon.transform.position, 0.01f, 0f);
    }


    /// <summary>
    /// Reorient to a random location within the room.
    /// </summary>
    /// <param name="beacon"></param>
    void UnattendFrom(GameObject beacon)
    {
        // TODO (still not quite right)
        Vector3 direction = Vector3.forward * Random.Range(-5f, 5f) + Vector3.right * Random.Range(-4f, 4f);
        transform.forward = Vector3.RotateTowards(transform.forward, direction, 0.01f, 0f);
    }


    Vector3 Bound(Vector3 position)
    {
        if (position.x < -4f) position.x = -4f + Random.Range(-0.01f, 0.01f);
        if (position.x > 4f) position.x = 4f + Random.Range(-0.01f, 0.01f);
        if (position.z < -5f) position.z = -5f + Random.Range(-0.01f, 0.01f);
        if (position.z > 5f) position.z = 5f + Random.Range(-0.01f, 0.01f);
        return position;

    }


    float Distance2D(Vector3 start, Vector3 end)
    {
        float d;
        float x = start.x - end.x;
        float z = start.z - end.z;
        d = x * x + z * z;
        return d;
    }
}
