using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

public class Agent : MonoBehaviour
{
    
    public float visualDistance = 4f;
    public float motorDistance = 1f;
    public float socialDistance = 1f;

    public float selfAttentionWeight = 0.5f;
    public float jointAttentionWeight = 0.5f;

    public float visualWeight;
    public float motorWeight;
    public float socialWeight;

    public Vector3 velocity;
    public Vector3 acceleration;
    public float maxForce = 2f;
    public float maxSpeed = 4f;


    void Awake()
    {
        CheckWeights();

        transform.localPosition = new Vector3(
            Random.Range(-4f, 4f), 0f, Random.Range(-5f, 5f)
        );
        
        float unitVelocity = Random.value * Mathf.PI * 2;
        velocity = new Vector3(Mathf.Cos(unitVelocity), 0f, Mathf.Sin(unitVelocity));
        //velocity = Vector3.zero;
        acceleration = Vector3.zero;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        UpdateVelocity();
        // For visualization only
        // Debug.DrawRay(transform.position, transform.forward);
    }


    public virtual void UpdateVelocity()
    {
        transform.localPosition += velocity * Time.deltaTime;   // x = v dt
        velocity += acceleration * Time.deltaTime;              // v = a dt
        if (velocity.magnitude > maxSpeed)
            velocity = velocity.normalized * maxSpeed;
        // transform.forward += velocity;
        acceleration *= 0f;
    }


    /// <summary>
    /// Combine all agent-based components for acceleration updating.
    /// </summary>
    /// <param name="agentGroup">Group of all neighboring agents</param>
    public virtual void FlockWith(List<Agent> agentGroup)
    {
        Vector3 alignment = Align(agentGroup, visualDistance);
        Vector3 cohesion = Amass(agentGroup, motorDistance);
        Vector3 separation = Avoid(agentGroup, socialDistance);

        acceleration += alignment;
        acceleration += cohesion;
        acceleration += separation;
    }


    /// <summary>
    /// Implements the alignment rule.
    /// </summary>
    /// <param name="agentGroup">All neighboring agents</param>
    /// <param name="visualDistance">Distance threshold for alignment</param>
    /// <returns>
    /// dir     : Vector3
    ///     3D motion vector component for influence of alignment.
    /// </returns>
    public virtual Vector3 Align(List<Agent> agentGroup, float visualDistance = 2f)
    {
        Vector3 dir = Vector3.zero;
        int neighbors = 0;

        foreach (Agent agent in agentGroup)
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
        return jointAttentionWeight * visualWeight * dir;
    }


    /// <summary>
    /// Implements the cohesion rule.
    /// </summary>
    /// <param name="agentGroup">All neighboring agents</param>
    /// <param name="motorDistance">Distance threshold for cohesion</param>
    /// <returns>
    /// dir     : Vector3
    ///     3D motion vector component for influence of cohesion.
    /// </returns>
    public virtual Vector3 Amass(List<Agent> agentGroup, float motorDistance = 2f)
    {
        Vector3 dir = Vector3.zero;
        int neighbors = 0;

        foreach (Agent agent in agentGroup)
        {
            Vector3 agentPosition = agent.transform.localPosition;
            float distance = Vector3.Distance(transform.localPosition, agentPosition);

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
        return jointAttentionWeight * motorWeight * dir;
    }


    /// <summary>
    /// Implements the separation rule.
    /// </summary>
    /// <param name="agentGroup">All neighboring agents</param>
    /// <param name="socialDistance">Distance threshold for separation</param>
    /// <returns>
    /// dir     : Vector3
    ///     3D motion vector component for influence of separation.
    /// </returns>
    public virtual Vector3 Avoid(List<Agent> agentGroup, float socialDistance = 0.5f)
    {
        Vector3 dir = Vector3.zero;
        int neighbors = 0;

        foreach (Agent agent in agentGroup)
        {
            Vector3 agentPosition = agent.transform.localPosition;
            float distance = Vector3.Distance(transform.localPosition, agentPosition);

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
        return jointAttentionWeight * socialWeight * dir;
    }


    /// <summary>
    /// Define boundary condition for the agent.
    /// </summary>
    public virtual void Bound()
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
    


    public virtual Vector3 Bounded(Vector3 position)
    {
        if (position.x < -4f) position.x = -4f + Random.Range(-0.01f, 0.01f);
        if (position.x > 4f) position.x = 4f + Random.Range(-0.01f, 0.01f);
        if (position.z < -5f) position.z = -5f + Random.Range(-0.01f, 0.01f);
        if (position.z > 5f) position.z = 5f + Random.Range(-0.01f, 0.01f);
        return position;
    }


    public virtual void CheckWeights()
    {
        Debug.LogAssertion(selfAttentionWeight + jointAttentionWeight == 1f);
        Debug.LogAssertion(visualWeight + motorWeight + socialWeight == 1f);
    }

}
