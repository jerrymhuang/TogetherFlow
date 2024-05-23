using UnityEngine;

public static class RNG
{
    static bool hasSpare = false;
    static float spare;

    public static float Gaussian(float mean = 0f, float std = 1f)
    {
        if (hasSpare)
        {
            hasSpare = false;
            return spare * std + mean;
        }

        float u, v, s;

        do
        {
            u = Random.value * 2f - 1f;
            v = Random.value * 2f - 1f;
            s = u * u + v * v;
        }
        while (s >= 1f || s == 0f);

        float scale = Mathf.Sqrt(-2f * Mathf.Log(s) / s);
        spare = scale * v;
        hasSpare = true;
        return mean + std * u * scale;
    }

    public static Vector3 Gaussian3D(Vector3 mean = default(Vector3), Vector3 std = default(Vector3))
    {
        if (mean == default(Vector3)) 
            mean = Vector3.zero;
        if (std == Vector3.zero || std == default(Vector3)) 
            std = Vector3.one;

        return new Vector3(
            Gaussian(mean.x, std.x), 
            Gaussian(mean.y, std.y),
            Gaussian(mean.z, std.z)
        );
    }
}
