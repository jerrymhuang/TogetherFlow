using System.Collections.Generic;
using UnityEngine;


public class Agent : MonoBehaviour
{

    [Header("Distance Thresholds")]
    public float visualDistance = 1f;
    public float motorDistance = 2.5f;
    public float socialDistance = 0.5f;

    public float visualWeight = 1f;
    public float motorWeight = 1f;
    public float socialWeight = 1f;

    [Header("Physical Properties")]
    public float maxForce = 2f;
    public float maxSpeed = 4f;

    // Component vectors of joint attention
    [HideInInspector]
    public Vector3 alignment;

    [HideInInspector]
    public Vector3 cohesion;

    [HideInInspector]
    public Vector3 separation;


    // Agent motion vectors
    [HideInInspector]
    public Vector3 velocity;

    [HideInInspector]
    public Vector3 acceleration;

    bool visualizeAgentProximity = false;
    bool visualizeVisualDistance = false;
    bool visualizeMotorDistance = false;
    bool visualizeSocialDistance = false;

    LineRenderer proximityVisualizer;

    void Awake()
    {
        // Check if the weights add up to be 1
        CheckWeights();

        // Initialize an agent position
        InitializePosition();

        // Initialize distance thresholds
        InitializeDistance();

        // Initialize agent motion vectors
        InitializeMotionVector();

        // Initialize agent proximity visualizer
        InitializeProximityVisualizer();

    }


    void Update()
    {
        Visualize();
    }

    void FixedUpdate()
    {
        // Update velocity motion vector
        UpdateVelocity();


    }


    public virtual void InitializePosition()
    {
        transform.localPosition = new Vector3(
            Random.Range(-4f, 4f), 0f, Random.Range(-5f, 5f)
        );
    }


    /// <summary>
    /// Sample agent distance thresholds in hierarchical manner.
    /// </summary>
    public virtual void InitializeDistance()
    {
        visualDistance = Random.Range(4f, 8f);
        motorDistance = Random.Range(2f, 8f);
        socialDistance = Random.Range(1f, 8f);
    }


    public virtual void InitializeMotionVector()
    {
        float unitVelocity = Random.value * Mathf.PI * 2;
        velocity = new Vector3(
            Mathf.Cos(unitVelocity), 0f, Mathf.Sin(unitVelocity)
        );
        // velocity = Vector3.zero;
        acceleration = Vector3.zero;
    }


    public virtual void InitializeProximityVisualizer()
    {
        proximityVisualizer = gameObject.AddComponent<LineRenderer>();
        proximityVisualizer.enabled = false;
        proximityVisualizer.startWidth = 0.25f;
        proximityVisualizer.loop = true;
    }


    /// <summary>
    /// Update velocity based upon the calculation of 
    /// </summary>
    public virtual void UpdateVelocity()
    {
        // x = v dt
        transform.localPosition += velocity * Time.deltaTime;   
        transform.localPosition = Bounded(transform.localPosition);

        // v = a dt
        velocity += acceleration * Time.deltaTime;
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
        // Update joint attention components
        alignment = Align(agentGroup, visualDistance);
        cohesion = Amass(agentGroup, motorDistance);
        separation = Avoid(agentGroup, socialDistance);
        
        // Add to acceleration
        acceleration += alignment;
        acceleration += cohesion;
        acceleration += separation;
    }


    /// <summary>
    /// Implements the alignment rule.
    /// </summary>
    /// <param name="agentGroup">All neighboring agents</param>
    /// <param name="visualDistance">Distance threshold for alignment</param>
    /// <param name="visualize">Optional boolean to visualize</param>
    /// <returns>
    /// dir     : Vector3
    ///     3D motion vector component for influence of alignment.
    /// </returns>
    public virtual Vector3 Align(
        List<Agent> agentGroup, 
        float visualDistance = 2f
    )
    {
        Vector3 dir = Vector3.zero;
        int neighbors = 0;

        foreach (Agent agent in agentGroup)
        {
            Vector3 agentPosition = agent.transform.localPosition;
            float distance = 
                Vector3.Distance(transform.localPosition, agentPosition);

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

        return visualWeight * dir;
    }


    /// <summary>
    /// Implements the cohesion rule.
    /// </summary>
    /// <param name="agentGroup">All neighboring agents</param>
    /// <param name="motorDistance">Distance threshold for cohesion</param>
    /// <param name="visualize">Optional boolean to visualize</param>
    /// <returns>
    /// dir     : Vector3
    ///     3D motion vector component for influence of cohesion.
    /// </returns>
    public virtual Vector3 Amass(
        List<Agent> agentGroup, 
        float motorDistance = 5f
    )
    {
        Vector3 dir = Vector3.zero;
        int neighbors = 0;

        foreach (Agent agent in agentGroup)
        {
            Vector3 agentPosition = agent.transform.localPosition;
            float distance = 
                Vector3.Distance(transform.localPosition, agentPosition);

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

        return motorWeight * dir;
    }


    /// <summary>
    /// Implements the separation rule.
    /// </summary>
    /// <param name="agentGroup">All neighboring agents</param>
    /// <param name="socialDistance">Distance threshold for separation</param>
    /// <param name="visualize">Optional boolean to visualize</param>
    /// <returns>
    /// dir     : Vector3
    ///     3D motion vector component for influence of separation.
    /// </returns>
    public virtual Vector3 Avoid(
        List<Agent> agentGroup, 
        float socialDistance = 1f
    )
    {
        Vector3 dir = Vector3.zero;
        int neighbors = 0;

        foreach (Agent agent in agentGroup)
        {
            Vector3 agentPosition = agent.transform.localPosition;
            float distance = 
                Vector3.Distance(transform.localPosition, agentPosition);

            if (distance < socialDistance && agent != this)
            {
                Vector3 socialPosition = 
                    transform.position - agent.transform.position;
                socialPosition = socialPosition / socialDistance;

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

        return socialWeight * dir;
    }
    
    /// <summary>
    /// Bound the agent within the room.
    /// </summary>
    /// <param name="position">Unbounded agent position</param>
    /// <returns>
    /// position        : Vector3
    ///     Bounded agent position
    /// </returns>
    public virtual Vector3 Bounded(Vector3 position)
    {
        if (position.x < -4f) position.x = -4f + Random.Range(-0.01f, 0.01f);
        if (position.x > 4f) position.x = 4f + Random.Range(-0.01f, 0.01f);
        if (position.z < -5f) position.z = -5f + Random.Range(-0.01f, 0.01f);
        if (position.z > 5f) position.z = 5f + Random.Range(-0.01f, 0.01f);
        
        if (position.y != 0f) position.y = 0f;
        
        return position;
    }


    /// <summary>
    /// Calculate distance between two points in the XZ plane
    /// </summary>
    /// <param name="start">Starting point</param>
    /// <param name="end">Ending point</param>
    /// <returns>
    /// d       : float
    ///     Distance between starting point and end point.
    /// </returns>
    public float Distance2D(Vector3 start, Vector3 end)
    {
        float d;
        float x = start.x - end.x;
        float z = start.z - end.z;
        d = Mathf.Sqrt(x * x + z * z);

        return d;
    }


    /// <summary>
    /// Check if the weights sum up to 1.
    /// If not, a warning is logged.
    /// </summary>
    public virtual void CheckWeights()
    {
        if (visualWeight + motorWeight + socialWeight - 1f > 0.00001f)
            Debug.LogWarning(
                "Elements of joint attention do not sum up to 1."
            );
    }


    public virtual void Visualize()
    {
        if (Input.GetKeyDown(KeyCode.V))
            visualizeVisualDistance = !visualizeVisualDistance;
        if (Input.GetKeyDown(KeyCode.M))
            visualizeMotorDistance = !visualizeMotorDistance;
        if (Input.GetKeyDown(KeyCode.N))
            visualizeSocialDistance = !visualizeSocialDistance;
        if (Input.GetKeyDown(KeyCode.B))
            visualizeAgentProximity = !visualizeAgentProximity;

        if (visualizeVisualDistance)
            VisualizeDistance(visualDistance, new Color(128f / 255f, 0f, 0f));
        if (visualizeMotorDistance)
            VisualizeDistance(motorDistance, new Color(78f / 255f, 42f / 255f, 132f / 155f));
        if (visualizeSocialDistance)
            VisualizeDistance(socialDistance, new Color(62f / 255f, 177f / 255f, 200f / 255f));
        if (visualizeAgentProximity)
            VisualizeAgentProximity(proximityVisualizer);

    }


    /// <summary>
    /// Visualize the distance setting for any agent as a circle.
    /// </summary>
    /// <param name="d">Distance setting</param>
    /// <param name="c">Color used for visualization</param>
    /// <param name="segments"># of segments for the visualized circle.</param>
    public virtual void VisualizeDistance(float d, Color c, int segments = 24)
    {
        Vector3 pos = transform.position;
        float a0, a1;
        Vector3 a, b;

        for (int i = 0; i < segments; i++)
        {
            a0 = Mathf.PI * 2f * i / segments;
            a1 = Mathf.PI * 2f * (i + 1) / segments;
            a = new Vector3(Mathf.Cos(a0), 0f, Mathf.Sin(a0)) * d;
            b = new Vector3(Mathf.Cos(a1), 0f, Mathf.Sin(a1)) * d;

            Debug.DrawLine(pos + a, pos + b, c);
        }
    }


    /// <summary>
    /// Visualize the direction of the agent
    /// </summary>
    /// <param name="dir">Direction to visualize</param>
    /// <param name="c">Color for the visualization</param>
    public virtual void VisualizeDirection(Vector3 dir, Color c)
    {
        Debug.DrawRay(transform.position, Vector3.Normalize(dir), c);
    }


    public virtual void VisualizeAgentProximity(
        LineRenderer proximityRenderer, 
        int vertexCount = 24,
        float personalRadius = 0.5f
    )
    {
        proximityRenderer.enabled = true;
        proximityRenderer.positionCount = vertexCount;

        Vector3[] vertices = new Vector3[vertexCount];

        for (int i = 0; i < vertexCount; i++)
        {
            float a = Mathf.PI * 2f * i / vertexCount;
            vertices[i] = transform.position + new Vector3(
                Mathf.Cos(a), 0f, Mathf.Sin(a)
            ) * personalRadius;
        }

        proximityRenderer.SetPositions(vertices);
    }


}
