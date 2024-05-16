using System.Linq;
using System.Collections.Generic;
using UnityEngine;


public class SampleAgent : MonoBehaviour
{
    /* This is a MonoBehaviour script, for now. 
     * Later, this will become inherited from Agent. */
    GameObject room;
    GameObject[] targets;
    float[] distances;

    int closest, current;
    bool move;

    private void Start()
    {
        transform.gameObject.tag = "Agent";
        Initialize();
    }


    private void Update()
    {
        Vector3 roomPosition = room.transform.position;
        closest = FindClosestTarget();
        if (closest != current)
        {
            move = true;

        }

        if (move)
        {
            transform.position = Vector3.MoveTowards(transform.position, targets[closest].transform.position, 0.05f);
            if (Mathf.Abs(roomPosition.x - transform.position.x) > 4f ||
                Mathf.Abs(roomPosition.z - transform.position.z) > 5f)
            {
                move = false;
                current = closest;
            }
        }
        // Debug.Log(closest);


            
        Debug.Log(Vector3.Distance(roomPosition, transform.position));
    }

    void Initialize()
    {
        room = GameObject.FindGameObjectWithTag("Room");
        targets = GameObject.FindGameObjectsWithTag("Target");
        distances = new float[targets.Length];
    }

    void WalkWithin()
    {

    }

    void WalkAround()
    { 

    }

    int FindClosestTarget()
    {
        int closestTarget = -1;
        float closestDistance;
        for (int i = 0; i < targets.Length; i++)
        {
            distances[i] = Vector3.Distance(transform.position, targets[i].transform.position);
        }

        closestDistance = Mathf.Min(distances);

        for (int i = 0; i < targets.Length; i++)
        {
            if (distances[i] == closestDistance)
            {
                closestTarget = i;
                break;
            }
        }
        return closestTarget;
    }

}
