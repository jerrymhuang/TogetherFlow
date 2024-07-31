using UnityEngine;

public class VizTest : MonoBehaviour
{
    LineRenderer lineRenderer;
    Vector3[] vertices;
    int numSegments = 24;

    void Start()
    {
        lineRenderer = GetComponent<LineRenderer>();
        lineRenderer.positionCount = numSegments;
        lineRenderer.loop = true;
        vertices = new Vector3[numSegments];
    }

    // Update is called once per frame
    void Update()
    {
        for (int i = 0; i < numSegments; i++)
        {
            float a = Mathf.PI * 2f * i / numSegments;
            vertices[i] = new Vector3(Mathf.Cos(a), 0f, Mathf.Sin(a));
            Debug.Log(vertices[i]);
        }

        lineRenderer.SetPositions(vertices);
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

}
