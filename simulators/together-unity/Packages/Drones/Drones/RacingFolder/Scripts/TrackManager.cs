using UnityEngine;
using EasyButtons;
/// <summary>
/// Handles the creation and management of the waypoints the participant flies though, collectively making a track
/// </summary>
public class TrackManager : MonoBehaviour
{
    #region Waypoint Management
    public delegate void TrackManagerStatus();
    public static TrackManagerStatus onWaypointPassed;
    public static TrackManagerStatus onTrackManagerFinished;


    public int WaypointCounter { get; private set; } = 0;
    [SerializeField]
    private GatelessTrackManager gatelessManager;
    [SerializeField]
    private ExperimentModeManager experimentModeManager;
    [Tooltip("Array of our waypoints, they get dynamically added and you can re-position them in order if something goes wrong.(Probably will not)")]
    public WaypointProperties[] waypointArray;
    [Tooltip("Is skipping waypoints allowed?")]
    public bool unordered;
    public GateProgressData GateData { get; private set; }

    public GameObject roundGateContainer;
    public GameObject lookaheadGateContainer;
    //public bool findRoundGate; //if true, populate waypointArray with RoundGates // if false, populate waypointArray with LookaheadGates
    //public bool findLookaheadGate;


    private static TrackManager _instance;
    public static TrackManager Instance { get { return _instance; } }
    private void Awake()
    {
        if (_instance != null && _instance != this)
        {
            Destroy(this.gameObject);
        }
        else
        {
            _instance = this;
            
        }
        //check if we have anywaypoints
        /*if (transform.childCount == 0)
            gameObject.SetActive(false);*/
    }

    private void Start()
    {
        FindWayPoints(); //finds allwaypoints if the inital list is 0, must be childed to this manager
        ToggleTriggers(unordered);
        LightRangeUpdate();
        WaypointSoundLevel();
        int lapCount = ExperimentManager.EXManager == null ? 2 : ExperimentManager.EXManager.CurrentBlock.numLaps;
        int numGates = 0;
        if (experimentModeManager.getCondition() != (ExperimentalCondition)6 && experimentModeManager.getCondition() != (ExperimentalCondition)7/*roundGateContainer.activeInHierarchyfindRoundGate*/)
        {
            numGates = roundGateContainer.GetComponentsInChildren<WaypointProperties>().Length;
        }
        else if (experimentModeManager.getCondition() == (ExperimentalCondition)6 || experimentModeManager.getCondition() == (ExperimentalCondition)7/*lookaheadGateContainer.activeInHierarchy!findRoundGate*/)
        {
            numGates = lookaheadGateContainer.GetComponentsInChildren<WaypointProperties>().Length;
        }
        GateData = new GateProgressData(lapCount, numGates);
    }


    public void FindWayPoints()
    {
        if (experimentModeManager.getCondition() != (ExperimentalCondition)6 && experimentModeManager.getCondition() != (ExperimentalCondition)7/*roundGateContainer.activeInHierarchyfindRoundGate*/)
        {
            waypointArray = roundGateContainer.GetComponentsInChildren<WaypointProperties>();
            for (int i = 0; i < waypointArray.Length; ++i) waypointArray[i].detector.index = i;
        }
        else if(experimentModeManager.getCondition() == (ExperimentalCondition)6 || experimentModeManager.getCondition() == (ExperimentalCondition)7/*lookaheadGateContainer.activeInHierarchy!findRoundGate*/)
        {
            waypointArray = lookaheadGateContainer.GetComponentsInChildren<WaypointProperties>();
            for (int i = 0; i < waypointArray.Length; ++i) waypointArray[i].detector.index = i;
        }
    }

    void ToggleTriggers(bool enabled)
    {
        foreach (WaypointProperties w in waypointArray)
        {
            w.detector.gameObject.SetActive(enabled);
            if (enabled)
            {
                MeshRenderer meshRenderer_NeonPipes = w.neonPipes.GetComponent<MeshRenderer>();
                meshRenderer_NeonPipes.materials[0].SetColor("_Color", activeWaypointColor);
                meshRenderer_NeonPipes.materials[0].SetColor("_EmissionColor", activeWaypointColor);
                //paint lights
                foreach (Transform lightChild in w.lights.transform)
                {
                    lightChild.GetComponent<Light>().color = activeWaypointColor;
                }
            }
        }

        ActivateNextWayPoint();
    }

    public void CollidedWithThisPoint(int waypointIndex)
    {
        GateData.TryRecordCollision(GatelessTrackManager.Instance.CurrentLap, waypointIndex);
    }

    public void PassedThroughThisPoint(int waypointIndex)
    {
        GateData.TryRecordPassage(GatelessTrackManager.Instance.CurrentLap, waypointIndex);
        if (unordered)
        {
            ActivatePreviousWayPoint();
            WaypointCounter = waypointIndex;
            DeactivateCurrentWayPoint();
            if (waypointIndex == waypointArray.Length - 1) WaypointCounter = 0;
            else WaypointCounter = waypointIndex + 1;
        }

        else
        {
            DeactivateCurrentWayPoint();
            // NEW LAP == Next trial or move on
            if (WaypointCounter == waypointArray.Length - 1)
            {
                WaypointCounter = 0;
                if (ExperimentManager.EXManager != null)
                {
                    if (!ExperimentManager.EXManager.InLastTrial())
                    {
                        ExperimentManager.Session.EndCurrentTrial();
                        ExperimentManager.Session.BeginNextTrial();
                    }
                    else
                    {
                        ExperimentManager.EXManager.LoadBlockAfterTrack();
                    }
                }
            }
            else
            {
                WaypointCounter++;
            }

            //LapTimerMethod();
            ActivateNextWayPoint();
        }

        onWaypointPassed?.Invoke();
    }

    [Tooltip("Color of waypoints that will be emmitting when they are not the next in the row to be passed through.")]
    public Color inactiveWaypointColor;
    [Tooltip("Color of single next waypoint that is second to come.")]
    public Color nextWaypointColor;
    [Tooltip("Color of current waypoint we need to pass through.")]
    public Color activeWaypointColor;
    [Tooltip("Range of light emitting from waypoints.")]
    [Range(0, 20)]
    public float lightRange = 10;
    [Tooltip("Volume of sound effect when passing through waypoint.")]
    [Range(0.0f, 1.0f)]
    public float waypointSound = 0.2f;

    /// <summary>
    /// Looks for a transform by name among a waypoint's children and it's children's children
    /// </summary>
    public static Transform FindInChildren(string name, Transform waypoint)
    {
        Transform t = waypoint.Find(name);
        
        if(t == null)
        {
            for (int child = 0; child < waypoint.childCount; ++child)
            {
                t = waypoint.GetChild(child).Find(name);
                
                if (t != null) break;
                
            }
        }
        if (t == null) Debug.Log("Total Failure");

        return t;
    }
    void ActivateNextWayPoint()
    {
        //activate trigger
        waypointArray[WaypointCounter].detector.gameObject.SetActive(true);

        //paint current waypoint gate
        MeshRenderer meshRenderer_NeonPipes = waypointArray[WaypointCounter].neonPipes.GetComponent<MeshRenderer>();
        meshRenderer_NeonPipes.materials[0].SetColor("_Color", activeWaypointColor);
        meshRenderer_NeonPipes.materials[0].SetColor("_EmissionColor", activeWaypointColor);
        //paint lights
        foreach (Transform lightChild in waypointArray[WaypointCounter].lights.transform)
        {
            if(lightChild.TryGetComponent(out Light l))
                l.color = activeWaypointColor;
        }


        //actiavte seoncd waypoint and paint
        if(!unordered) ActivateSecondNextWayPoint();
    }

    public void LightRangeUpdate()
    {
        foreach (WaypointProperties t in waypointArray)
        {
            foreach (Transform lightChild in t.lights.transform)
            {
                if(lightChild.TryGetComponent(out Light l))
                    l.range = lightRange;
            }
        }
    }

    public void WaypointSoundLevel()
    {
        foreach (WaypointProperties t in waypointArray)
        {
            if(t.sound.TryGetComponent(out AudioSource audio))
                audio.volume = waypointSound;

        }
    }


    void ActivateSecondNextWayPoint()
    {
        int secondNextWayPoint = 0;
        if (WaypointCounter + 1 == waypointArray.Length)
        {
            secondNextWayPoint = 0;
        }
        else
        {
            secondNextWayPoint = WaypointCounter + 1;
        }

        //deactivate trigger
        waypointArray[secondNextWayPoint].detector.gameObject.SetActive(false);

        //paint current waypoint gate
        MeshRenderer meshRenderer_NeonPipes = waypointArray[secondNextWayPoint].neonPipes.GetComponent<MeshRenderer>();
        meshRenderer_NeonPipes.materials[0].SetColor("_Color", nextWaypointColor);
        meshRenderer_NeonPipes.materials[0].SetColor("_EmissionColor", nextWaypointColor);
        //paint lights
        foreach (Transform lightChild in waypointArray[secondNextWayPoint].lights.transform)
        {
            lightChild.GetComponent<Light>().color = nextWaypointColor;
        }

    }

    void DeactivateCurrentWayPoint()
    {
        //deactivate trigger
        waypointArray[WaypointCounter].detector.gameObject.SetActive(false);

        //paint current waypoint gate
        MeshRenderer meshRenderer_NeonPipes = waypointArray[WaypointCounter].neonPipes.GetComponent<MeshRenderer>();
        meshRenderer_NeonPipes.materials[0].SetColor("_Color", inactiveWaypointColor);
        meshRenderer_NeonPipes.materials[0].SetColor("_EmissionColor", inactiveWaypointColor);
        //paint lights
        foreach (Transform lightChild in waypointArray[WaypointCounter].lights.transform)
        {
            lightChild.GetComponent<Light>().color = inactiveWaypointColor;
        }
    }

    void ActivatePreviousWayPoint()
    {
        int prevWaypoint = 0;
        if (WaypointCounter - 1 < 0 )
        {
            prevWaypoint = waypointArray.Length - 1;
        }
        else
        {
            prevWaypoint = WaypointCounter - 1;
        }

        //deactivate trigger
        waypointArray[prevWaypoint].detector.gameObject.SetActive(true);

        //paint current waypoint gate
        MeshRenderer meshRenderer_NeonPipes = waypointArray[prevWaypoint].neonPipes.GetComponent<MeshRenderer>();
        meshRenderer_NeonPipes.materials[0].SetColor("_Color", activeWaypointColor);
        meshRenderer_NeonPipes.materials[0].SetColor("_EmissionColor", activeWaypointColor);
        //paint lights
        foreach (Transform lightChild in waypointArray[prevWaypoint].lights.transform)
        {
            lightChild.GetComponent<Light>().color = activeWaypointColor;
        }
    }

    //IM A BIG Batman FAN - "Anakin Skywalker"
    //p.s. don't feed the troll
    void OnePaintMethodToRuleThemAll()
    {

    }
    #endregion


    #region Waypoint Generation
    [Space][Space]
    [Tooltip("Recommended: WaypointLoop.")]
    public GameObject waypointPrefab1;
    [Tooltip("Recommended: WaypointFlag.")]
    public GameObject waypointPrefab2;
    [Tooltip("Recommended: WaypointRegular.")]
    public GameObject waypointPrefab3;
    public GameObject waypointContainer;

    public GameObject CreateWaypoint(GameObject waypointPrefab)
    {
        GameObject newWaypoint = Instantiate(waypointPrefab, waypointContainer.transform);
        newWaypoint.transform.rotation = Quaternion.Euler(Vector3.zero);
        //Set the location at the most recently placed waypoint
        if (waypointContainer.transform.childCount > 1)
            newWaypoint.transform.localPosition = waypointContainer.transform.GetChild(waypointContainer.transform.childCount - 2).localPosition;
        else
            newWaypoint.transform.localPosition = Vector3.zero;

        return newWaypoint;
    }

    private void OnDrawGizmos()
    {
        foreach (WaypointProperties t in waypointArray)
        {
            if(t != null)
            {
                if (t.p0 != null && t.p0.activeSelf == true)
                {
                    Gizmos.color = new Color(1, 0, 0, 0.6f);
                    Gizmos.DrawSphere(t.p0.transform.position, 0.5f);
                }
                if (t.pm0 != null && t.pm0.activeSelf == true)
                {
                    Gizmos.color = new Color(1, 0, 0, 0.6f);
                    Gizmos.DrawSphere(t.pm0.transform.position, 0.5f);
                }
                if (t.p1 != null && t.p1.activeSelf == true)
                {
                    Gizmos.color = new Color(1, 0, 0, 0.6f);
                    Gizmos.DrawSphere(t.p1.transform.position, 0.5f);
                }
            }

        }
    }

    #endregion


}