using UnityEngine;

public class SampleTarget : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Awake()
    {
        transform.gameObject.tag = "Target";
    }

}
