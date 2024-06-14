using UnityEngine;
using System.Linq;

public class LocomotiveAgent : MonoBehaviour
{

    private GameObject beacon;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        transform.localPosition = Vector3.right * Random.Range(-4f, 4f) + 
                                  Vector3.forward * Random.Range(-5f, 5f);
        beacon = GameObject.FindGameObjectWithTag("Beacon");
    }


    // Update is called once per frame
    void Update()
    {
        WalkTo(beacon);
    }


    void WalkTo(GameObject beacon)
    {

    }

}
