using UnityEngine;

public class WaypointDetector : MonoBehaviour {

    [HideInInspector]
    public int index;

    public Collider[] MyColliders { get; private set; }

    private AudioSource passSound;

    private void Awake()
    {
        try
        {
            passSound = transform.parent.Find("sound").GetComponent<AudioSource>();
        }
        catch(System.Exception ex)
        {
            Debug.LogError("Did not found sound gameobject. -> " + ex);
        }

        MyColliders = GetComponents<Collider>(); //getting the box collider componente to turn this to trigger so we dont have to do it manually
        foreach(Collider c in MyColliders)
        {
            c.isTrigger = true;
        }
    }

    //checks for entring trigger
    private void OnTriggerEnter(Collider other)
    {
        if(TrackManager.Instance && other.transform.tag == "Player")//searches for the player tag in the root of the transform (all drones must be player tags)
        {
            if (passSound) passSound.Play();
            TrackManager.Instance.PassedThroughThisPoint(index); //let track manager known which waypoint we just passed(returning parent of detector transform because that one is stored in the array)
            this.transform.parent.transform.parent.GetComponent<TargetUpdater>().CallUpdateTargets();
        }
    }

}
