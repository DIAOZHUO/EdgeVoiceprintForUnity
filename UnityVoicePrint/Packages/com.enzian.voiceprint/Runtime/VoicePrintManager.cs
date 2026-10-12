using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityVAD;

namespace VoicePrint
{
    /// <summary>
    /// Scene entry point for voice-print enrollment / identification.
    ///
    /// * Loads VoicePrintEngine (ECAPA + Silero VAD on ONNX Runtime) off the main thread.
    /// * Optionally listens to UnityVAD.AudioRecorder: every speech segment it cuts from the
    ///   microphone is enrolled (Mode.Enroll) or identified (Mode.Identify).
    /// * Async API for your own audio: EnrollAsync / IdentifyAsync (float[] 16 kHz or AudioClip).
    ///   Inference runs on a worker thread, one job at a time; results and events come back on
    ///   the main thread.
    /// * Preset Speakers (name + AudioClips) are enrolled at startup, before Loaded fires.
    /// * The speaker library persists as JSON in Application.persistentDataPath.
    /// </summary>
    public class VoicePrintManager : MonoBehaviour
    {
        /// The first VoicePrintManager in the scene (like UnityVAD.AudioRecorder.Instance), set in Awake.
        public static VoicePrintManager Instance;

        public enum Mode { Identify, Enroll, Off }

        [Serializable]
        public class PresetSpeaker
        {
            public string name;
            [Tooltip("Clean single-speaker clips, >= 3 s of speech in total. Import as Decompress On Load + PCM " +
                     "(Vorbis re-encoding measurably shifts the voice print).")]
            public AudioClip[] clips;
        }

        [Header("Models")]
        [Tooltip("ECAPA-TDNN ONNX as .bytes (Assets/VoicePrint/Models/ecapa_tdnn_int8.bytes)")]
        public TextAsset ecapaModel;
        [Tooltip("silero_vad.bytes from the Silero VAD for Unity package. Leave empty to run without VAD.")]
        public TextAsset vadModel;
        [Tooltip("Optional emb_mean.bytes. Only used when Center Embedding is on.")]
        public TextAsset embMean;
        public bool centerEmbedding = false;
        [Range(1, 8)] public int intraOpThreads = 2;

        [Header("Pipeline")]
        public VoicePrintSettings settings = new VoicePrintSettings();

        [Header("Microphone (UnityVAD.AudioRecorder)")]
        public bool listenToAudioRecorder = true;
        public Mode mode = Mode.Identify;
        [Tooltip("Speaker name used for microphone segments while Mode = Enroll.")]
        public string enrollName = "Speaker";

        [Header("Preset enrollment")]
        [Tooltip("Enrolled at startup from these clips. A preset replaces any saved print with the same " +
                 "name (it is rebuilt from its clips every run); other saved speakers are kept.")]
        public List<PresetSpeaker> presetSpeakers = new List<PresetSpeaker>();

        [Header("Identification")]
        [Tooltip("Tag passed to OnSpeechIdentified when the best match is below the accept threshold / margin, " +
                 "or no speaker is enrolled.")]
        public string unknownTag = "Unknown";

        [Header("Library")]
        public bool persistLibrary = true;
        public string libraryFileName = "voiceprint_library.json";

        public event Action Loaded;                            // models loaded and presets enrolled
        public event Action<string, EmbedResult> Enrolled;     // (speaker, result) also fires for rejected clips
        public event Action<IdentifyResult> Identified;

        /// <summary>
        /// Fired for identified speech, on the main thread. audio = clean VAD speech (mono 16 kHz),
        /// tag = accepted speaker name or unknownTag. Not fired for non-speech.
        /// * IdentifySegmentAsync / microphone segments (settings.maxPieceSeconds > 0): once per speaker
        ///   turn -- consecutive pieces with the same tag are concatenated, so a segment where two
        ///   people talk yields one clip per person per turn.
        /// * IdentifyAsync: once for the whole clip.
        /// </summary>
        public event SpeechIdentifiedHandler OnSpeechIdentified;
        public delegate void SpeechIdentifiedHandler(float[] audio, string tag);
        public event Action LibraryChanged;

        public bool IsReady => ready && engine != null;
        public bool IsBusy => pending > 0;
        public VoicePrintEngine Engine => engine;
        public SpeakerLibrary Library => engine?.Library;
        public string LibraryPath => Path.Combine(Application.persistentDataPath, libraryFileName);

        VoicePrintEngine engine;
        bool ready;
        readonly SemaphoreSlim inferenceGate = new SemaphoreSlim(1, 1);
        int pending;
        AudioRecorder recorder;

        void Awake()
        {
            if (Instance == null) Instance = this;
        }

        async void Start()
        {
            if (ecapaModel == null)
            {
                Debug.LogError("[VoicePrint] ecapaModel is not assigned.");
                return;
            }
            var ecapaBytes = ecapaModel.bytes;   // TextAsset access must stay on the main thread
            var vadBytes = vadModel != null ? vadModel.bytes : null;
            var mean = centerEmbedding && embMean != null ? VectorMath.FromBytes(embMean.bytes) : null;
            if (vadBytes == null && settings.useVad)
                Debug.LogWarning("[VoicePrint] vadModel not assigned; running without VAD gating.");

            try
            {
                var e = await Task.Run(() => new VoicePrintEngine(ecapaBytes, vadBytes, settings, mean, intraOpThreads));
                if (this == null) { e.Dispose(); return; }   // destroyed while loading
                engine = e;
            }
            catch (Exception ex)
            {
                Debug.LogError("[VoicePrint] failed to load models: " + ex);
                return;
            }

            LoadLibrary();
            await EnrollPresetsAsync();
            if (engine == null) return;                        // destroyed while enrolling presets
            ready = true;
            Debug.Log($"[VoicePrint] ready. VAD={(engine.HasVad ? "on" : "off")}, {engine.Library.Count} enrolled speaker(s).");
            Loaded?.Invoke();

            if (listenToAudioRecorder)
            {
                recorder = AudioRecorder.Instance != null ? AudioRecorder.Instance : FindAnyObjectByType<AudioRecorder>();
                if (recorder != null) recorder.MIC_SegmentAddedEventHandler += OnMicSegment;
                else Debug.LogWarning("[VoicePrint] listenToAudioRecorder is on but no AudioRecorder is in the scene.");
            }
        }

        void OnDestroy()
        {
            if (Instance == this) Instance = null;
            if (recorder != null) recorder.MIC_SegmentAddedEventHandler -= OnMicSegment;
            if (engine == null) return;
            // let an in-flight inference finish before freeing the native sessions
            if (inferenceGate.Wait(5000))
            {
                engine.Dispose();
                inferenceGate.Release();
            }
            engine = null;
        }

        async void OnMicSegment(VADSegment segment)
        {
            if (mode == Mode.Off || !IsReady) return;
            int rate = (int)recorder.samplingRate;
            var wav = AudioUtil.ToMono16k(segment.Audio, 1, rate);
            // the streaming VAD's probabilities line up with the audio only at 16 kHz
            var probs = rate == LogMelFbank.SampleRate ? segment.ChunkProbs : null;
            try
            {
                if (mode == Mode.Enroll) await EnrollAsync(enrollName, wav);
                else if (settings.maxPieceSeconds > 0f) await IdentifySegmentAsync(wav, probs);
                else await IdentifyAsync(wav);
            }
            catch (Exception ex)
            {
                Debug.LogError("[VoicePrint] " + ex);
            }
        }

        // ---------------------------------------------------------------- public API

        public async Task<EmbedResult> EnrollAsync(string speaker, AudioClip clip) =>
            await EnrollAsync(speaker, await AudioUtil.ToMono16kAsync(clip));

        public async Task<IdentifyResult> IdentifyAsync(AudioClip clip) =>
            await IdentifyAsync(await AudioUtil.ToMono16kAsync(clip));

        /// <summary>
        /// (Re)build every preset speaker from its clips, replacing same-name prints.
        /// Called automatically at startup; call again after editing presetSpeakers at runtime.
        /// </summary>
        public async Task EnrollPresetsAsync()
        {
            bool changed = false;
            foreach (var preset in presetSpeakers)
            {
                if (preset == null || string.IsNullOrWhiteSpace(preset.name) || preset.clips == null) continue;
                if (engine == null) return;
                changed |= engine.Library.Remove(preset.name);
                int accepted = 0;
                float seconds = 0f;
                foreach (var clip in preset.clips)
                {
                    if (clip == null) continue;
                    try
                    {
                        var wav = await AudioUtil.ToMono16kAsync(clip);
                        var r = await Run(e => e.Enroll(preset.name, wav));
                        if (r.IsSpeech)
                        {
                            accepted++;
                            seconds += r.UsedSamples / (float)LogMelFbank.SampleRate;
                            changed = true;
                        }
                        else Debug.LogWarning($"[VoicePrint] preset '{preset.name}': clip '{clip.name}' rejected as non-speech.");
                        Enrolled?.Invoke(preset.name, r);
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning($"[VoicePrint] preset '{preset.name}': clip '{clip.name}' failed: {ex.Message}");
                    }
                }
                if (accepted == 0) Debug.LogWarning($"[VoicePrint] preset '{preset.name}' has no usable clips; not enrolled.");
                else if (seconds < 3f) Debug.LogWarning($"[VoicePrint] preset '{preset.name}' has only {seconds:F1}s of speech; >= 3 s is recommended.");
            }
            if (changed)
            {
                SaveLibrary();
                LibraryChanged?.Invoke();
            }
        }

        /// <summary>Add a mono 16 kHz clip to the speaker's voice print.</summary>
        public async Task<EmbedResult> EnrollAsync(string speaker, float[] wav16k)
        {
            var r = await Run(e => e.Enroll(speaker, wav16k));
            if (r.IsSpeech)
            {
                SaveLibrary();
                LibraryChanged?.Invoke();
            }
            Enrolled?.Invoke(speaker, r);
            return r;
        }

        /// <summary>Identify the speaker of a mono 16 kHz clip against the library.</summary>
        public async Task<IdentifyResult> IdentifyAsync(float[] wav16k)
        {
            var r = await Run(e => e.Identify(wav16k));
            Identified?.Invoke(r);
            if (r.IsSpeech) OnSpeechIdentified?.Invoke(r.Embed.SpeechAudio, TagOf(r));
            return r;
        }

        /// <summary>
        /// Identify a segment that may contain several speakers: it is split into pieces of at most
        /// settings.maxPieceSeconds, each piece is identified (Identified fires per piece) and the
        /// speaker turns are reported through OnSpeechIdentified. Returns the pieces in time order.
        /// vadProbs: optional per-512-sample streaming VAD probabilities (UnityVAD.VADSegment.ChunkProbs).
        /// </summary>
        public async Task<List<SpeechPiece>> IdentifySegmentAsync(float[] wav16k, float[] vadProbs = null)
        {
            var pieces = await Run(e => e.IdentifySegment(wav16k, vadProbs));
            string turnTag = null;
            var turn = new List<float>();
            foreach (var p in pieces)
            {
                if (!p.Result.IsSpeech) continue;               // too short to embed
                Identified?.Invoke(p.Result);
                string tag = TagOf(p.Result);
                if (turnTag != null && tag != turnTag)
                {
                    OnSpeechIdentified?.Invoke(turn.ToArray(), turnTag);
                    turn.Clear();
                }
                turnTag = tag;
                turn.AddRange(p.Audio);
            }
            if (turnTag != null) OnSpeechIdentified?.Invoke(turn.ToArray(), turnTag);
            return pieces;
        }

        /// <summary>Accepted speaker name, otherwise unknownTag (null for non-speech).</summary>
        public string TagOf(IdentifyResult r) => !r.IsSpeech ? null : r.Accepted ? r.Speaker : unknownTag;


        public void RemoveSpeaker(string speaker)
        {
            if (engine != null && engine.Library.Remove(speaker))
            {
                SaveLibrary();
                LibraryChanged?.Invoke();
            }
        }

        public void ClearLibrary()
        {
            if (engine == null) return;
            engine.Library.Clear();
            SaveLibrary();
            LibraryChanged?.Invoke();
        }

        // ---------------------------------------------------------------- internals

        async Task<T> Run<T>(Func<VoicePrintEngine, T> job)
        {
            if (engine == null) throw new InvalidOperationException("VoicePrintManager is not ready yet (models loading).");
            Interlocked.Increment(ref pending);
            try
            {
                await inferenceGate.WaitAsync();
                try
                {
                    var e = engine;   // null once OnDestroy has disposed it
                    if (e == null) throw new ObjectDisposedException(nameof(VoicePrintManager));
                    return await Task.Run(() => job(e));
                }
                finally { inferenceGate.Release(); }
            }
            finally
            {
                Interlocked.Decrement(ref pending);
            }
        }

        void LoadLibrary()
        {
            if (!persistLibrary || !File.Exists(LibraryPath)) return;
            try
            {
                engine.Library.LoadJson(File.ReadAllText(LibraryPath));
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[VoicePrint] could not read {LibraryPath}: {ex.Message}");
            }
        }

        void SaveLibrary()
        {
            if (!persistLibrary || engine == null) return;
            try
            {
                File.WriteAllText(LibraryPath, engine.Library.ToJson());
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[VoicePrint] could not write {LibraryPath}: {ex.Message}");
            }
        }
    }
}
