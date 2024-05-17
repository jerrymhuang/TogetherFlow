using UnityEngine;


public class WaypointeRendererBuilder : MonoBehaviour {

    private WaypointRenderer waypointRenderer;
    private TrackManager trackManager;

    void Update() {
        if (!waypointRenderer) {
            waypointRenderer = GetComponent<WaypointRenderer>();
        }
        else
        {
            waypointRenderer.FindLineRenderer();
            waypointRenderer.InitalizeLists();
            waypointRenderer.FindChildrenAndWaypointAnchors();
            waypointRenderer.DrawQuadraticCurve();
        }

        if (!trackManager)
        {
            trackManager = FindObjectOfType<TrackManager>();
        }
      
	}

    void OnDisable()
    {
        gameObject.GetComponent<WaypointeRendererBuilder>().enabled = true;
    }
}
