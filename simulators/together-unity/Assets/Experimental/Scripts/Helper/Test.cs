using UnityEngine;

public class Test : MonoBehaviour
{

    // Update is called once per frame
    void FixedUpdate()
    {
        /* transform.forward = <cos(z), sin(z)> = <sin(x), cos(x)> */
        Debug.Log(Time.deltaTime + " " + transform.forward);
    }
}
