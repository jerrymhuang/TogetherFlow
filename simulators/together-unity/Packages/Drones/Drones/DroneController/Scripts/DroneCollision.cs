using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class DroneCollision : MonoBehaviour {

    #region PUBLIC VARIABLES

    [Tooltip("Sparks GameObject prefab that will be created when crashing the drone.")]
    public GameObject sparks;

    #endregion

    #region Mono Behaviour METHODS

    private List<GameObject> CollisionStack = new List<GameObject>();
    private void OnCollisionEnter(Collision collision)
    {
        if (!CollisionStack.Contains(collision.gameObject))
        {
            CollisionStack.Add(collision.gameObject);
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        CollisionStack.Remove(collision.gameObject);
    }

    public string GetCurrentCollision()
    {
        if (CollisionStack.Count > 0)
        {
            if (CollisionStack[CollisionStack.Count - 1].TryGetComponent(out GazeTarget target))
            {
                return target.Name();
            }

            else return CollisionStack[CollisionStack.Count - 1].name;
        }

        else return AirSimUnity.PositionRecorder.STR_NULL;
    }

    void OnCollisionStay(Collision other)
    {
        if (other.transform)
        {
            ContactPoint contact = other.contacts[0];
            Quaternion rot = Quaternion.FromToRotation(Vector3.up, contact.normal) * Quaternion.Euler(-90, 0, 0);
            Vector3 pos = contact.point;

            if (sparks)
            {
                GameObject spark = (GameObject)Instantiate(sparks, pos, rot);
                spark.transform.localScale = transform.localScale * 2;
                foreach (Transform _spark in spark.transform)
                {
                    _spark.localScale = transform.localScale * 2;
                }
                StartCoroutine(SparksCleaner(spark));
            }


        }
    }

    #endregion

    #region PRIVATE Coroutine METHODS

    IEnumerator SparksCleaner(GameObject _spark)
    {
        yield return new WaitForSeconds(1);
        Destroy(_spark);
    }

    #endregion

}
