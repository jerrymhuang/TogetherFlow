using System.Collections.Generic;
using UnityEngine;


public class FlockingAgent : MonoBehaviour
{
    /* Implement 2D agent with flocking behaviors only, 
     * mainly as a benchmark but also as a way of understanding ABMs.
     */

    /* This is a MonoBehaviour script, for now. 
     * Later, this will become inherited from a more general Agent class. 
     */

    [HideInInspector]
    public float visualDistance = 4f;

    [SerializeField]
    float motorDistance = 1f;

    [SerializeField]
    float socialDistance = 1f;

    [HideInInspector]
    public Vector3 velocity;

    [HideInInspector]
    Vector3 acceleration;

    [HideInInspector]
    float maxForce = 2f;

    [HideInInspector]
    float maxSpeed = 4f;

    void Awake()
    {
        transform.localPosition = new Vector3(
            Random.Range(-4f, 4f), 0f, Random.Range(-5f, 5f)
        );

        float unitVelocity = Random.value * Mathf.PI * 2;
        velocity = new Vector3(Mathf.Cos(unitVelocity), 0f, Mathf.Sin(unitVelocity));
        acceleration = Vector3.zero;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        transform.localPosition += velocity * Time.deltaTime;
        velocity += acceleration * Time.deltaTime;
        if (velocity.magnitude > maxSpeed) 
            velocity = velocity.normalized * maxSpeed;
        transform.forward = velocity;
        acceleration *= 0f;

        // For visualization only
        Debug.DrawRay(transform.position, transform.forward);
    }


    public void FlockWith(List<FlockingAgent> agentGroup)
    {
        Vector3 alignment = Align(agentGroup, visualDistance);
        Vector3 cohesion = Amass(agentGroup, motorDistance);
        Vector3 separation = Avoid(agentGroup, socialDistance);
        acceleration += alignment;
        acceleration += cohesion;
        acceleration += separation;
    }


    public Vector3 Align(List<FlockingAgent> agentGroup, float visualDistance = 2f)
    {
        Vector3 dir = Vector3.zero;
        int neighbors = 0;

        foreach (FlockingAgent agent in agentGroup)
        {
            Vector3 agentPosition = agent.transform.localPosition;
            float distance = Vector3.Distance(transform.localPosition, agentPosition);

            if (distance < visualDistance && agent != this)
            {
                dir += agent.velocity;
                neighbors++;
            }

            if (neighbors > 0)
            {
                dir /= neighbors;
                dir = dir.normalized * maxSpeed;
                dir -= velocity;
                if (dir.magnitude > maxForce) dir = dir.normalized * maxForce;
            }
        }
        return dir;
    }


    public Vector3 Amass(List<FlockingAgent> agentGroup, float motorDistance = 2f)
    {
        Vector3 dir = Vector3.zero;
        int neighbors = 0;

        foreach (FlockingAgent agent in agentGroup)
        {
            Vector3 agentPosition = agent.transform.localPosition;
            float distance = Vector3.Distance(transform.localPosition, agentPosition);
            //Debug.Log(distance);

            if (distance < motorDistance && agent != this)
            {
                dir += agent.transform.localPosition;
                neighbors++;
            }

            if (neighbors > 0)
            {
                dir /= neighbors;
                dir -= transform.localPosition;
                dir = dir.normalized * maxSpeed;
                dir -= velocity;
                if (dir.magnitude > maxForce) dir = dir.normalized * maxForce;
            }
        }
        return dir;
    }


    public Vector3 Avoid(List<FlockingAgent> agentGroup, float socialDistance = 0.5f)
    {
        Vector3 dir = Vector3.zero;
        int neighbors = 0;

        foreach (FlockingAgent agent in agentGroup)
        {
            Vector3 agentPosition = agent.transform.localPosition;
            float distance = Vector3.Distance(transform.localPosition, agentPosition);
            //Debug.Log(distance);

            if (distance < socialDistance && agent != this)
            {
                Vector3 socialPosition = 
                    (transform.position - agent.transform.position) / socialDistance;

                dir += socialPosition;
                neighbors++;
            }

            if (neighbors > 0)
            {
                dir /= neighbors;
                dir = dir.normalized * maxSpeed;
                dir -= velocity;
                if (dir.magnitude > maxForce) dir = dir.normalized * maxForce;
            }
        }
        return dir;
    }


    public void Bound()
    {
        Vector3 pos = transform.localPosition;

        if (transform.localPosition.x < -4f)
        {
            pos.x = -4f;
            velocity = Vector3.Reflect(velocity, Vector3.right);
        }

        if (transform.localPosition.x >= 4f)
        {
            pos.x = 4f;
            velocity = Vector3.Reflect(velocity, Vector3.left);
        }

        if (transform.localPosition.z <= -5f)
        {
            pos.z = -5f;
            velocity = Vector3.Reflect(velocity, Vector3.forward);
        }

        if (transform.localPosition.z >= 5f)
        {
            pos.z = 5f;
            velocity = Vector3.Reflect(velocity, Vector3.back);
        }

        transform.localPosition = pos;

    }
}
