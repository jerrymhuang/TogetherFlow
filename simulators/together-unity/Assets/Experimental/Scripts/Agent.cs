using System.Collections.Generic;
using UnityEngine;


public class Agent : MonoBehaviour
{

    [Header("Distance Thresholds")]
    public float visualDistance = 4f;
    public float motorDistance = 1f;
    public float socialDistance = 1f;

    [Header("Weights")]
    public float selfAttentionWeight = 0f;
    public float jointAttentionWeight = 1f;

    public float visualWeight = 0.1f;
    public float motorWeight = 0.1f;
    public float socialWeight = 0.8f;

    [Header("Physical Properties")]
    public float maxForce = 2f;
    public float maxSpeed = 4f;


    [HideInInspector]
    public Vector3 alignment;

    [HideInInspector]
    public Vector3 cohesion;

    [HideInInspector]
    public Vector3 separation;

    [HideInInspector]
    public Vector3 velocity;

    [HideInInspector]
    public Vector3 acceleration;


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
        Bound();
        velocity += acceleration * Time.deltaTime;              // v = a dt
        if (velocity.magnitude > maxSpeed)
            velocity = velocity.normalized * maxSpeed;
        transform.forward += velocity;
        acceleration *= 0f;
    }


    /// <summary>
    /// Combine all agent-based components for acceleration updating.
    /// </summary>
    /// <param name="agentGroup">Group of all neighboring agents</param>
    public virtual void FlockWith(List<Agent> agentGroup)
    {
        alignment = Align(agentGroup, visualDistance);
        cohesion = Amass(agentGroup, motorDistance);
        separation = Avoid(agentGroup, socialDistance, true);


        acceleration += alignment;
        acceleration += cohesion;
        acceleration += separation;

        Debug.Log(transform.gameObject.name + ": " + acceleration);
        Debug.DrawRay(transform.position, Vector3.Normalize(acceleration), Color.red);
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
    public virtual Vector3 Align(List<Agent> agentGroup, float visualDistance = 2f, bool visualize = false)
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

        if (visualize) VisualizeDistance(visualDistance, Color.cyan);
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
    public virtual Vector3 Amass(List<Agent> agentGroup, float motorDistance = 2f, bool visualize = false)
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

        if (visualize) VisualizeDistance(motorDistance, Color.yellow);
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
    public virtual Vector3 Avoid(List<Agent> agentGroup, float socialDistance = 0.5f, bool visualize = false)
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

        if (visualize) VisualizeDistance(socialDistance, Color.magenta, 24);
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

        if (pos.y != 0f) pos.y = 0f;

        transform.localPosition = pos;
    }
    


    public virtual Vector3 Bounded(Vector3 position)
    {
        if (position.x < -4f) position.x = -4f + Random.Range(-0.01f, 0.01f);
        if (position.x > 4f) position.x = 4f + Random.Range(-0.01f, 0.01f);
        if (position.z < -5f) position.z = -5f + Random.Range(-0.01f, 0.01f);
        if (position.z > 5f) position.z = 5f + Random.Range(-0.01f, 0.01f);
        
        if (position.y != 0f) position.y = 0f;
        
        return position;
    }


    public float Distance2D(Vector3 start, Vector3 end)
    {
        float d;
        float x = start.x - end.x;
        float z = start.z - end.z;
        d = Mathf.Sqrt(x * x + z * z);

        return d;
    }


    public virtual void CheckWeights()
    {
        if (selfAttentionWeight + jointAttentionWeight != 1f)
            Debug.LogWarning("Self-attention and joint attention do not sum up to 1.");
        if (visualWeight + motorWeight + socialWeight - 1f > 0.00001f)
            Debug.LogWarning("Elements of joint attention do not sum up to 1.");
    }


    public virtual void VisualizeDistance(float d, Color c, int segments = 12)
    {
        Vector3 pos = transform.position;
        float a0, a1;
        Vector3 a, b;

        for (int i = 0; i < segments; i++)
        {
            a0 = Mathf.PI * 2f * i / segments;
            a1 = Mathf.PI * 2f * (i + 1) / segments;
            a = (Vector3.right * Mathf.Cos(a0) + Vector3.forward * Mathf.Sin(a0)) * d;
            b = (Vector3.right * Mathf.Cos(a1) + Vector3.forward * Mathf.Sin(a1)) * d;

            Debug.DrawLine(pos + a, pos + b, c);
        }
    }
}
