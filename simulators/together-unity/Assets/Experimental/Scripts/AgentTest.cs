using UnityEngine;

public class AgentTest : MonoBehaviour
{

    [SerializeField] 
    GameObject obj;

    float t = 0f;
    float T = 3f;

    float attentionSwitchingThreshold = 0.5f;

    bool count = true;
    bool attend = true;





    private void Start()
    {
        Debug.Log(obj.transform.position - transform.position);

    }

    void Update()
    {
        // TestAttend();
        TestAttentionSwitching();
    }


    void TestAttend()
    {
        attend = Random.Range(0f, 1f) < attentionSwitchingThreshold;

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
        }

        if (Vector3.Angle(transform.forward, Vector3.right) == 0f)
        {
            count = false;
        }

        if (count)
        {
            t += Time.deltaTime;
        }
    }


    void TestAttentionSwitching()
    {
        // If a beacon is switched, then attention goes back to 1.


        // Otherwise, attention decreases to 0 nonlinearly.


        // When unttend, the agent random-walks to anywhere in the room.


        // A beacon is attended to when it is within a certain distance
        // from the agent.
    }

}
