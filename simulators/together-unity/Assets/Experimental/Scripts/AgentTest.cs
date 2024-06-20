using System.Collections;
using Unity.VisualScripting;
using UnityEngine;

public class AgentTest : MonoBehaviour
{

    [SerializeField] GameObject obj;

    float t = 0f;
    float T = 3f;

    bool count = false;


    private void Start()
    {
        Debug.Log(obj.transform.position - transform.position);

    }

    void Update()
    {
        TestAttend();
        Debug.Log(t);
    }


    void TestAttend()
    {
        Vector3 relativePosition = obj.transform.position - transform.position;
        
        if (t <= T)
        {
            // Start orienting towards the target
            transform.forward = Vector3.RotateTowards(
                transform.forward, relativePosition, 0.02f, 1f
            );

        }

        else if (t > T)
        {
            transform.forward = Vector3.RotateTowards(
                transform.forward, Vector3.right, 0.02f, 1f
            );
        }

        // When the agent is oriented towards the target, start the countdown
        if (Vector3.Angle(transform.forward, relativePosition) == 0f)
        {
            count = true;
            t += Time.deltaTime;
        }

        if (Vector3.Angle(transform.forward, Vector3.right) == 0f)
        {
            count = false;
            t = 0f;
        }

    }

}
