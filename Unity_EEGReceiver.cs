using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using Newtonsoft.Json;

/// <summary>
/// Unity script to receive EEG data from the EEG Bridge Python script
/// 
/// Usage:
/// 1. Attach this script to a GameObject in your scene
/// 2. Configure the UDP port (default 5005)
/// 3. Start the EEG Bridge script: python eeg_bridge.py --run --user yourname
/// 4. The OnEEGDataReceived event will fire with each update
/// 
/// Example integration:
/// - Use relax_index/focus_index to control game difficulty
/// - Use state ("relax"/"focus"/"neutral") for discrete game mechanics
/// - Use mi_prob_left/mi_prob_right for left/right hand motor imagery controls
/// </summary>
public class EEGReceiver : MonoBehaviour 
{
    [Header("Network Configuration")]
    public int udpPort = 5005;
    public bool logMessages = true;
    
    [Header("EEG Data (Read-only)")]
    [SerializeField] private EEGData currentData;
    
    // Events for other scripts to subscribe to
    public static event Action<EEGData> OnEEGDataReceived;
    public static event Action<string> OnStateChanged;  // "relax", "focus", "neutral"
    
    private UdpClient udpClient;
    private bool isReceiving = false;
    private string lastState = "";
    
    void Start() 
    {
        StartReceiving();
    }
    
    void OnDestroy() 
    {
        StopReceiving();
    }
    
    void StartReceiving() 
    {
        try 
        {
            udpClient = new UdpClient(udpPort);
            udpClient.BeginReceive(ReceiveCallback, null);
            isReceiving = true;
            
            Debug.Log($"EEG Receiver started on port {udpPort}");
        }
        catch (Exception e) 
        {
            Debug.LogError($"Failed to start EEG receiver: {e.Message}");
        }
    }
    
    void StopReceiving() 
    {
        if (isReceiving && udpClient != null) 
        {
            isReceiving = false;
            udpClient.Close();
            Debug.Log("EEG Receiver stopped");
        }
    }
    
    void ReceiveCallback(IAsyncResult ar) 
    {
        if (!isReceiving) return;
        
        try 
        {
            IPEndPoint ip = new IPEndPoint(IPAddress.Any, udpPort);
            byte[] bytes = udpClient.EndReceive(ar, ref ip);
            string message = Encoding.UTF8.GetString(bytes);
            
            // Parse JSON on main thread
            ProcessEEGMessage(message);
            
            // Continue receiving
            if (isReceiving) 
            {
                udpClient.BeginReceive(ReceiveCallback, null);
            }
        }
        catch (Exception e) 
        {
            if (isReceiving) 
            {
                Debug.LogError($"EEG receive error: {e.Message}");
                
                // Try to restart receiving
                try 
                {
                    udpClient.BeginReceive(ReceiveCallback, null);
                }
                catch 
                {
                    Debug.LogError("Failed to restart EEG receiving");
                }
            }
        }
    }
    
    void ProcessEEGMessage(string jsonMessage) 
    {
        try 
        {
            var eegData = JsonConvert.DeserializeObject<EEGData>(jsonMessage);
            
            // Update current data (for inspector display)
            currentData = eegData;
            
            // Log if enabled
            if (logMessages) 
            {
                Debug.Log($"EEG: State={eegData.state}, Relax={eegData.relax_index:F2}, " +
                         $"Focus={eegData.focus_index:F2}, MI L/R={eegData.mi_prob_left:F2}/{eegData.mi_prob_right:F2}");
            }
            
            // Fire events
            OnEEGDataReceived?.Invoke(eegData);
            
            // State change event
            if (eegData.state != lastState) 
            {
                lastState = eegData.state;
                OnStateChanged?.Invoke(eegData.state);
                Debug.Log($"EEG State changed to: {eegData.state}");
            }
        }
        catch (Exception e) 
        {
            Debug.LogError($"Failed to parse EEG JSON: {e.Message}\nMessage: {jsonMessage}");
        }
    }
    
    // Public methods for other scripts to access current data
    public EEGData GetCurrentData() 
    {
        return currentData;
    }
    
    public float GetRelaxIndex() 
    {
        return currentData?.relax_index ?? 0f;
    }
    
    public float GetFocusIndex() 
    {
        return currentData?.focus_index ?? 0f;
    }
    
    public string GetCurrentState() 
    {
        return currentData?.state ?? "neutral";
    }
    
    public float GetMotorImageryLeft() 
    {
        return currentData?.mi_prob_left ?? 0f;
    }
    
    public float GetMotorImageryRight() 
    {
        return currentData?.mi_prob_right ?? 0f;
    }
    
    public bool IsArtifact() 
    {
        return currentData?.artifact ?? false;
    }
}

/// <summary>
/// Data structure matching the JSON sent by EEG Bridge
/// </summary>
[Serializable]
public class EEGData 
{
    public float t;                    // Timestamp
    public float alpha;                // Alpha band power
    public float beta;                 // Beta band power  
    public float theta;                // Theta band power
    public float relax_index;          // Relaxation index (higher = more relaxed)
    public float focus_index;          // Focus index (higher = more focused)
    public string state;               // "relax", "focus", or "neutral"
    public float mi_prob_left;         // Motor imagery probability for left hand
    public float mi_prob_right;        // Motor imagery probability for right hand
    public bool artifact;              // True if data quality is poor
}

/// <summary>
/// Example usage script showing how to respond to EEG data
/// Attach this to any GameObject to see EEG integration examples
/// </summary>
public class EEGGameController : MonoBehaviour 
{
    [Header("Game Response Settings")]
    public float relaxThreshold = 1.2f;
    public float focusThreshold = 0.8f;
    public float motorImageryThreshold = 0.7f;
    
    [Header("Visual Feedback")]
    public Light ambientLight;         // Changes color based on state
    public GameObject leftHandObject;  // Activates with left MI
    public GameObject rightHandObject; // Activates with right MI
    
    void OnEnable() 
    {
        // Subscribe to EEG events
        EEGReceiver.OnEEGDataReceived += HandleEEGData;
        EEGReceiver.OnStateChanged += HandleStateChange;
    }
    
    void OnDisable() 
    {
        // Unsubscribe from events
        EEGReceiver.OnEEGDataReceived -= HandleEEGData;
        EEGReceiver.OnStateChanged -= HandleStateChange;
    }
    
    void HandleEEGData(EEGData data) 
    {
        // Skip if artifact detected
        if (data.artifact) 
        {
            return;
        }
        
        // Handle motor imagery
        HandleMotorImagery(data);
        
        // Handle continuous values (could control game parameters)
        HandleContinuousValues(data);
    }
    
    void HandleStateChange(string newState) 
    {
        // Change ambient lighting based on mental state
        if (ambientLight != null) 
        {
            switch (newState) 
            {
                case "relax":
                    ambientLight.color = Color.blue;
                    break;
                case "focus":
                    ambientLight.color = Color.red;
                    break;
                default:
                    ambientLight.color = Color.white;
                    break;
            }
        }
        
        // You could also:
        // - Change game difficulty based on focus state
        // - Trigger special abilities during relaxed state
        // - Pause game during neutral state
        Debug.Log($"Game responding to state: {newState}");
    }
    
    void HandleMotorImagery(EEGData data) 
    {
        // Left hand motor imagery
        bool leftActive = data.mi_prob_left > motorImageryThreshold;
        if (leftHandObject != null) 
        {
            leftHandObject.SetActive(leftActive);
        }
        
        // Right hand motor imagery
        bool rightActive = data.mi_prob_right > motorImageryThreshold;
        if (rightHandObject != null) 
        {
            rightHandObject.SetActive(rightActive);
        }
        
        // Example: Move player character
        if (leftActive && !rightActive) 
        {
            // Move left
            transform.Translate(Vector3.left * Time.deltaTime);
        }
        else if (rightActive && !leftActive) 
        {
            // Move right
            transform.Translate(Vector3.right * Time.deltaTime);
        }
    }
    
    void HandleContinuousValues(EEGData data) 
    {
        // Example: Use focus index to control game speed
        float gameSpeedMultiplier = Mathf.Lerp(0.5f, 2.0f, data.focus_index / 2.0f);
        Time.timeScale = gameSpeedMultiplier;
        
        // Example: Use relax index to control health regeneration
        float healthRegen = data.relax_index * 0.1f;
        // Apply health regeneration...
        
        // Example: Use band powers for environmental effects
        // - Alpha waves could control ambient music volume
        // - Beta waves could control particle effects intensity
        // - Theta waves could influence dream-like visual effects
    }
}
