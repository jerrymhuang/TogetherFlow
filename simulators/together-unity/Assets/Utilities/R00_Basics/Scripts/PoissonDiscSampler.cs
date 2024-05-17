using System.Collections.Generic;
using UnityEngine;


public class PoissonDiscSampler
{
    public static List<Vector2> GenerateSamples(
        float radius, Vector2 sampleRegionSize, 
        int sampleRejectionThreshold = 30
    )
    {
        float cellSize = radius * Mathf.Sqrt(2);

        /* Initialize a grid. 
            * What it does: tells us for each cell what the index of the point
            * in the points list is. */
        int[,] grid = new int[
            Mathf.CeilToInt(sampleRegionSize.x / cellSize),
            Mathf.CeilToInt(sampleRegionSize.y / cellSize)
        ];


        List<Vector2> indices = new List<Vector2>();
        List<Vector2> samples = new List<Vector2>();

        /* Initialize the first sample as the center of the grid 
            * This can be anywhere you want.*/
        samples.Add(sampleRegionSize / 2);

        /* Generate the next samples */
        while (samples.Count > 0)
        {
            int sampleId = Random.Range(0, samples.Count);
            Vector2 sample = samples[sampleId];
            bool accepted = false;

            for (int i = 0; i < sampleRejectionThreshold; i++)
            {
                float angle = Random.value * Mathf.PI * 2;
                Vector2 dir = 
                    new Vector2(Mathf.Sin(angle), Mathf.Cos(angle));

                /* Create candidate around the sample as center */
                Vector2 candidate = 
                    sample + dir * Random.Range(radius, 2 * radius);

                if (IsValid(candidate, sampleRegionSize, 
                    cellSize, radius, indices, grid))
                {
                    indices.Add(candidate);
                    samples.Add(candidate);

                    /* Assign the point to a cell */
                    int x = (int)(candidate.x / cellSize),
                        y = (int)(candidate.y / cellSize);
                    // Debug.Log(x.ToString() + " | " + y.ToString());
                    grid[x, y] = indices.Count;
                    accepted = true;
                    break;
                }
            }
            if (!accepted)
            {
                samples.RemoveAt(sampleId);
            }
        }

        return indices;
    }

    static bool IsValid(
        Vector2 candidate, Vector2 sampleRegionSize, 
        float cellSize, float radius, List<Vector2> samples, int[,] grid
    )
    {
        if (candidate.x >= 0 && candidate.x < sampleRegionSize.x && 
            candidate.y >= 0 && candidate.y < sampleRegionSize.y)
        {
            int cellX = (int)(candidate.x / cellSize),
                cellY = (int)(candidate.y / cellSize),
                startX = Mathf.Max(0, cellX - 2),
                startY = Mathf.Max(0, cellY - 2),
                endX = Mathf.Min(cellX + 2, grid.GetLength(0) - 1),
                endY = Mathf.Min(cellY + 2, grid.GetLength(1) - 1);

            for (int x = startX; x <= endX; x++)
            {
                for (int y = startY; y <= endY; y++)
                {
                    int sampleId = grid[x, y] - 1;
                    if (sampleId != -1)
                    {
                        float squaredDistance = 
                            (candidate - samples[sampleId]).sqrMagnitude;
                        if (squaredDistance < radius * radius)
                        {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
        return false;
    }
}