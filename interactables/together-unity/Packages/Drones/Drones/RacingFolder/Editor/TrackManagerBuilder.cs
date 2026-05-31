using UnityEngine;
using UnityEditor;

/// <summary>
/// GUI script that adds necessary components to the TrackManger for track building. Three prefab slots are made available
/// used as the objects to spawn as part of the manager
/// </summary>
[CustomEditor(typeof(TrackManager))]
public class TrackManagerBuilder : Editor
{
    private int waypointTypes = 3;
    private bool useWaypointLights = true;
    private void OnEnable()
    {

        if (((TrackManager)target).waypointArray == null)
        {
            ((TrackManager)target).waypointArray = new WaypointProperties[waypointTypes];
        }
           
    }
    public void AddSpaces(int num)
    {
        for(int i = 0; i < num; ++i)
        {
            EditorGUILayout.Space();
        }
    }
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        TrackManager trackManager = (TrackManager)target;
        AddSpaces(2);

        #region CustomWaypoints
        //  EditorGUILayout.HelpBox("Hover NAME OF VARIABLES for properties to find out more about them. If you're not sure what they are used for feel free to contact me via e-mail or watch the youtube tutorials i prepared first.", MessageType.Info);
        EditorGUILayout.LabelField("USE THIS TO CREATE CUSTOM WAYPOINTS", EditorStyles.toolbarButton);
        //trackManager.waypointPrefab1 = (GameObject)EditorGUILayout.ObjectField(new GUIContent("Waypoint Prefab 1", "Recommended: WaypointRegular."), trackManager.waypointPrefab1, typeof(GameObject), true);
        string name = trackManager.waypointPrefab1 == null ? "NULL" : trackManager.waypointPrefab1.name;
        if (GUILayout.Button("Create " + name))
        {
            Selection.activeObject = trackManager.CreateWaypoint(trackManager.waypointPrefab1);
        }
        AddSpaces(2);
        //trackManager.waypointPrefab2 = (GameObject)EditorGUILayout.ObjectField(new GUIContent("Waypoint Prefab 2", "Recommended: WaypointLoop."), trackManager.waypointPrefab2, typeof(GameObject), true);
        string name2 = trackManager.waypointPrefab2 == null ? "NULL" : trackManager.waypointPrefab2.name;
        if (GUILayout.Button("Create " + name2))
        {
            Selection.activeObject = trackManager.CreateWaypoint(trackManager.waypointPrefab2);
        }
        AddSpaces(2);
        //trackManager.waypointPrefab3 = (GameObject)EditorGUILayout.ObjectField(new GUIContent("Waypoint Prefab 3", "Recommended: WaypointFlag."), trackManager.waypointPrefab3, typeof(GameObject), true);
        string name3 = trackManager.waypointPrefab3 == null ? "NULL" : trackManager.waypointPrefab3.name;
        if (GUILayout.Button("Create " + name3))
        {
            Selection.activeObject = trackManager.CreateWaypoint(trackManager.waypointPrefab3);
        }
        AddSpaces(2);
        #endregion


        AddSpaces(3);

        if (GUILayout.Button("Enable/Disable Waypoint Line"))
        {
            trackManager.gameObject.GetComponent<WaypointRenderer>().enabled = !trackManager.gameObject.GetComponent<WaypointRenderer>().enabled;
            trackManager.gameObject.GetComponent<LineRenderer>().enabled = !trackManager.gameObject.GetComponent<LineRenderer>().enabled;
            trackManager.gameObject.GetComponent<LineRendererShader>().enabled = !trackManager.gameObject.GetComponent<LineRendererShader>().enabled;
        }
        if(GUILayout.Button("Enable/Disable Waypoint Lights"))
        {
            foreach(WaypointProperties w in trackManager.GetComponentsInChildren<WaypointProperties>())
            {
                w.neonPipes.gameObject.SetActive(useWaypointLights);
                w.lights.gameObject.SetActive(useWaypointLights);
            }

            useWaypointLights = !useWaypointLights;
        }
    }


}
