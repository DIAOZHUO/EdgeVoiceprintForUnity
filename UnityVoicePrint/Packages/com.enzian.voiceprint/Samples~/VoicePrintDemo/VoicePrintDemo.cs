using System.Collections.Generic;
using UnityEngine;
using UnityVAD;

namespace VoicePrint
{
    /// <summary>
    /// uGUI front-end for VoicePrintManager + AudioRecorder. The UI is built in the demo scene
    /// (Screen Space Canvas, desktop / Editor); this script only wires listeners: UI controls ->
    /// manager settings / calls, manager + recorder events -> UI. For a Quest headset switch the
    /// Canvas to World Space.
    /// Enroll: type a name, set Mode to Enroll, speak >= 3 s. Identify: set Mode back and speak.
    /// </summary>
    public class VoicePrintDemo : MonoBehaviour
    {
        public VoicePrintManager manager;
        [Tooltip("Clips listed in the Test clips panel (Load Type: Decompress On Load).")]
        public AudioClip[] testClips;

        [Header("Status")]
        public UnityEngine.UI.Text statusText;
        public UnityEngine.UI.Image micIndicator;
        public Color speakingColor = new Color(0.20f, 0.75f, 0.30f);
        public Color silentColor = new Color(0.45f, 0.45f, 0.45f);

        [Header("Controls")]
        [Tooltip("Options in VoicePrintManager.Mode order: Identify, Enroll, Off.")]
        public UnityEngine.UI.Dropdown modeDropdown;
        public UnityEngine.UI.InputField enrollNameInput;
        public UnityEngine.UI.Button clearLibraryButton;

        [Header("Lists (rows are cloned from inactive templates inside each list)")]
        [Tooltip("Row with a child Text 'Label' and a child Button 'RemoveButton'.")]
        public GameObject speakerRowTemplate;
        [Tooltip("Row with a child Text 'Label' and child Buttons 'EnrollButton' and 'IdentifyButton'.")]
        public GameObject clipRowTemplate;

        [Header("Log")]
        public UnityEngine.UI.Text logText;
        public int maxLogLines = 40;

        readonly List<string> log = new List<string>();
        readonly List<UnityEngine.UI.Button> clipButtons = new List<UnityEngine.UI.Button>();
        AudioRecorder recorder;
        bool speaking;

        void Start()
        {
            if (manager == null)
                manager = VoicePrintManager.Instance != null ? VoicePrintManager.Instance : FindAnyObjectByType<VoicePrintManager>();

            // UI -> manager
            modeDropdown.SetValueWithoutNotify((int)manager.mode);
            modeDropdown.onValueChanged.AddListener(i => manager.mode = (VoicePrintManager.Mode)i);
            enrollNameInput.SetTextWithoutNotify(manager.enrollName);
            enrollNameInput.onValueChanged.AddListener(s => manager.enrollName = s);
            clearLibraryButton.onClick.AddListener(manager.ClearLibrary);
            BuildClipRows();

            // manager / recorder -> UI
            manager.Loaded += OnLoaded;
            manager.LibraryChanged += RefreshSpeakers;
            manager.Enrolled += OnEnrolled;
            manager.OnSpeechIdentified += OnSpeechIdentified;
            recorder = AudioRecorder.Instance != null ? AudioRecorder.Instance : FindAnyObjectByType<AudioRecorder>();
            if (recorder != null) recorder.MIC_SpeechEventHandler += OnMicSpeechEvent;

            SetInteractable(manager.IsReady);
            if (manager.IsReady) OnLoaded();
            UpdateStatus();
        }

        void OnDestroy()
        {
            if (manager != null)
            {
                manager.Loaded -= OnLoaded;
                manager.LibraryChanged -= RefreshSpeakers;
                manager.Enrolled -= OnEnrolled;
                manager.OnSpeechIdentified -= OnSpeechIdentified;
            }
            if (recorder != null) recorder.MIC_SpeechEventHandler -= OnMicSpeechEvent;
        }

        // IsBusy has no event; refresh the status line every frame
        void Update() => UpdateStatus();

        // ---------------------------------------------------------------- event handlers

        void OnLoaded()
        {
            SetInteractable(true);
            RefreshSpeakers();
            Log($"ready: {manager.Library.Count} enrolled speaker(s)");
        }

        void OnEnrolled(string speaker, EmbedResult r) => Log(r.IsSpeech
            ? $"enrolled '{speaker}' ({r.UsedSamples / 16000f:F1}s, {manager.Library.ClipCount(speaker)} clip(s))"
            : $"enroll '{speaker}' rejected: non-speech");

        void OnSpeechIdentified(float[] audio, string tag) => Log($"{tag}: {audio.Length / 16000f:F1}s");

        void OnMicSpeechEvent(SpeechEvent e) => speaking = e == SpeechEvent.Start;

        // ---------------------------------------------------------------- UI building

        void RefreshSpeakers()
        {
            var list = speakerRowTemplate.transform.parent;
            ClearRows(list, speakerRowTemplate);
            if (!manager.IsReady) return;
            foreach (var name in manager.Library.Names)
            {
                var row = Instantiate(speakerRowTemplate, list);
                row.SetActive(true);
                row.transform.Find("Label").GetComponent<UnityEngine.UI.Text>().text =
                    $"{name}  ×{manager.Library.ClipCount(name)}";
                string speaker = name;
                row.transform.Find("RemoveButton").GetComponent<UnityEngine.UI.Button>()
                    .onClick.AddListener(() => manager.RemoveSpeaker(speaker));
            }
        }

        void BuildClipRows()
        {
            var list = clipRowTemplate.transform.parent;
            ClearRows(list, clipRowTemplate);
            clipButtons.Clear();
            if (testClips == null) return;
            foreach (var clip in testClips)
            {
                if (clip == null) continue;
                var row = Instantiate(clipRowTemplate, list);
                row.SetActive(true);
                row.transform.Find("Label").GetComponent<UnityEngine.UI.Text>().text = clip.name;
                var c = clip;
                var enroll = row.transform.Find("EnrollButton").GetComponent<UnityEngine.UI.Button>();
                var identify = row.transform.Find("IdentifyButton").GetComponent<UnityEngine.UI.Button>();
                enroll.onClick.AddListener(() => _ = manager.EnrollAsync(manager.enrollName, c));
                identify.onClick.AddListener(() => _ = manager.IdentifyAsync(c));
                clipButtons.Add(enroll);
                clipButtons.Add(identify);
            }
        }

        static void ClearRows(Transform list, GameObject template)
        {
            for (int i = list.childCount - 1; i >= 0; i--)
            {
                var child = list.GetChild(i).gameObject;
                if (child != template) Destroy(child);
            }
        }

        void SetInteractable(bool on)
        {
            modeDropdown.interactable = on;
            enrollNameInput.interactable = on;
            clearLibraryButton.interactable = on;
            foreach (var b in clipButtons) b.interactable = on;
        }

        void UpdateStatus()
        {
            micIndicator.color = speaking ? speakingColor : silentColor;
            statusText.text = !manager.IsReady
                ? "Loading models / enrolling presets..."
                : $"Mic: {(speaking ? "speaking" : "silent")}{(manager.IsBusy ? "   [inferring]" : "")}";
        }

        void Log(string line)
        {
            log.Insert(0, $"[{System.DateTime.Now:HH:mm:ss}] {line}");
            if (log.Count > maxLogLines) log.RemoveAt(log.Count - 1);
            logText.text = string.Join("\n", log);
        }
    }
}
