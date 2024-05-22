using UnityEngine;

public class TimeStep : MonoBehaviour
{

    // Update is called once per frame
    void FixedUpdate()
    {
        Debug.Log(Time.deltaTime);
    }
}
