using UnityEngine;

public class Room : MonoBehaviour
{
    float angle;
    Vector3 dir;

    void Update()
    {
        angle = Mathf.PI * 0.5f - Mathf.Deg2Rad * transform.rotation.eulerAngles.y;
        dir = new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle));

        Debug.DrawRay(transform.position, dir, Color.magenta);
    }

}
