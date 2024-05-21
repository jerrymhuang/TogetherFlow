using UnityEngine;

public static class GaussianRNG
{
    static bool hasSpare = false;
    static float spare;

    public static float Sample(float mean = 0f, float std = 1f)
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
}
