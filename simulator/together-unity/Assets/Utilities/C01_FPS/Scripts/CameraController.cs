using Cinemachine;
using UnityEngine;

using EMPACResearch.Core.Singletons;

/// <summary>
/// class CameraController : Singleton<CameraController>
/// ===========================================================================
/// 
/// </summary>
public class CameraController : Singleton<CameraController> 
{
    [SerializeField]
    bool isAvatar;

    [SerializeField]
    float amplitudeGain = 0.5f, frequencyGain = 0.5f;

    [SerializeField]
    CinemachineVirtualCamera virtualCamera;

    private void Awake()
    {
        virtualCamera = GetComponent<CinemachineVirtualCamera>();
    }

    public void Follow(Transform transform)
    {
        virtualCamera.Follow = transform;

        if (isAvatar)
        {
            var perlin = 
                virtualCamera.GetCinemachineComponent<CinemachineBasicMultiChannelPerlin>();
            perlin.m_AmplitudeGain = amplitudeGain;
            perlin.m_FrequencyGain = frequencyGain;
        }
    }

}
