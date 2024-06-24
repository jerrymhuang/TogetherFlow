using UnityEngine;

/* class Singleton<T> : NetworkBehaviour
 * ============================================================================
 * Create a single instance that only lives within the lifecycle of a scene.
 * It will die when the scene is destroyed
 * (can be changed with DontDestroyOnLoad()).
 * It can inherit from both *MonoBehaviour (this script)* and NetworkBehaviour.
 */

namespace EMPACResearch.Core.Singletons
{
    public class Singleton<T> : MonoBehaviour
       where T : Component
    {
        private static T instance;
        public static T Instance
        {
            get
            {
                if (instance == null)
                {
                    var objs = FindObjectsByType(typeof(T), FindObjectsSortMode.None) as T[];
                    if (objs.Length > 0)
                    {
                        instance = objs[0];
                    }
                    if (objs.Length > 1)
                    {
                        Debug.LogError("More than one " + typeof(T).Name + " in the scene.");
                    }
                    if (instance == null)
                    {
                        GameObject obj = new GameObject();
                        obj.name = string.Format("{0}", typeof(T).Name);
                        instance = obj.AddComponent<T>();
                    }
                }
                return instance;
            }
        }
    }
}
