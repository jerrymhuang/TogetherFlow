using DroneController.Physics;
using UnityEngine;
using FunctionLibrary;

public enum EControlComplexity
{
    Simple, Complex
}

public class DroneMovement : DroneMovementScript {
    public Vector3 t_position;
    public Quaternion t_rotation;
    public Vector3 t_euler;
    public Transform dronePositionAnchor;
    public Transform droneRotationAnchor;
    
    [Tooltip("Simple Controls")]
    [Range(0, 1000)]
    public float targetHeight;
    public override void Awake()
    {

        base.Awake(); //I would suggest you to put code below this line or in a Start() method
    }

    private void Start()
    {
        if (ExperimentManager.useRCController)
        {
            customFeed = false;
            mobile_turned_on = false;
            joystick_turned_on = true;
        }

        else
        {
            customFeed = false;
            mobile_turned_on = false;
            joystick_turned_on = false;
        }
    }

    void FixedUpdate()
    {
        GetVelocity();
        ClampingSpeedValues();
        SettingControllerToInputSettings(); //sensitivity settings for joystick,keyboard,mobile (depending on which is turned on)
        if (FlightRecorderOverride == false)
        {
            if (!disableInput)
            {
                switch (profiles[_profileIndex].controlComplexity)
                {
                    case EControlComplexity.Complex:
                        MovementUpDown();
                        MovementLeftRight();
                        Rotation();
                        MovementForward();
                        break;
                    case EControlComplexity.Simple:
                        MovementLeftRight();
                        Rotation();
                        MovementForward();
                        dronePositionAnchor.position = dronePositionAnchor.position.SwapY(targetHeight);
                        break;
                }
            }
            BasicDroneHoverAndRotation(); //this method applies all the forces and rotations to the drone.
        }
    }


    void Update () {
        RotationUpdateLoop_TrickRotation(); //applies rotation to the drone it self when doing the barrel roll trick, does NOT trigger the animation
        Animations(); //part where animations are triggered
        DroneSound(); //sound producing stuff
        CameraCorrectPickAndTranslatingInputToWSAD(); //setting input for keys, translating joystick, mobile inputs as WSAD (depending on which is turned on)
        t_position = dronePositionAnchor.position;
        t_rotation = droneRotationAnchor.rotation;
        t_euler = droneRotationAnchor.eulerAngles;
    }
}
