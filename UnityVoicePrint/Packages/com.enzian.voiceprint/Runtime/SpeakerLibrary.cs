using System;
using System.Collections.Generic;
using UnityEngine;

namespace VoicePrint
{
    /// <summary>
    /// Enrolled voice prints. Each speaker keeps the running sum of its unit embeddings;
    /// the print is the re-normalized mean (centroid). Thread-safe; JSON round-trips via
    /// JsonUtility (ToJson / LoadJson).
    /// </summary>
    public sealed class SpeakerLibrary
    {
        [Serializable]
        class Entry
        {
            public string name;
            public int count;
            public float[] sum;
        }

        [Serializable]
        class Data
        {
            public int version = 1;
            public List<Entry> speakers = new List<Entry>();
        }

        readonly object gate = new object();
        readonly Dictionary<string, Entry> entries = new Dictionary<string, Entry>();
        readonly List<string> order = new List<string>();

        public int Count { get { lock (gate) return entries.Count; } }

        public IReadOnlyList<string> Names { get { lock (gate) return order.ToArray(); } }

        public int ClipCount(string name)
        {
            lock (gate) return entries.TryGetValue(name, out var e) ? e.count : 0;
        }

        public void Add(string name, float[] unitEmbedding)
        {
            if (string.IsNullOrEmpty(name)) throw new ArgumentException("speaker name is empty");
            lock (gate)
            {
                if (!entries.TryGetValue(name, out var e))
                {
                    e = new Entry { name = name, sum = new float[unitEmbedding.Length] };
                    entries[name] = e;
                    order.Add(name);
                }
                for (int i = 0; i < e.sum.Length; i++) e.sum[i] += unitEmbedding[i];
                e.count++;
            }
        }

        public bool Remove(string name)
        {
            bool removed;
            lock (gate)
            {
                removed = entries.Remove(name);
                if (removed) order.Remove(name);
            }
            return removed;
        }

        public void Clear()
        {
            lock (gate)
            {
                entries.Clear();
                order.Clear();
            }
        }

        /// <summary>Unit-length voice print of one speaker, or null.</summary>
        public float[] GetPrint(string name)
        {
            lock (gate)
                return entries.TryGetValue(name, out var e) ? VectorMath.Normalize((float[])e.sum.Clone()) : null;
        }

        /// <summary>(name, unit-length print) for every speaker, in enrollment order.</summary>
        public List<(string name, float[] print)> Snapshot()
        {
            lock (gate)
            {
                var list = new List<(string, float[])>(order.Count);
                foreach (var n in order) list.Add((n, VectorMath.Normalize((float[])entries[n].sum.Clone())));
                return list;
            }
        }

        public string ToJson(bool pretty = false)
        {
            var d = new Data();
            lock (gate)
                foreach (var n in order)
                {
                    var e = entries[n];
                    d.speakers.Add(new Entry { name = e.name, count = e.count, sum = (float[])e.sum.Clone() });
                }
            return JsonUtility.ToJson(d, pretty);
        }

        /// <summary>Replace the library with the contents of ToJson() output.</summary>
        public void LoadJson(string json)
        {
            var d = JsonUtility.FromJson<Data>(json);
            lock (gate)
            {
                entries.Clear();
                order.Clear();
                if (d?.speakers != null)
                    foreach (var e in d.speakers)
                    {
                        if (string.IsNullOrEmpty(e.name) || e.sum == null || e.sum.Length != SpeakerEncoder.EmbeddingDim) continue;
                        entries[e.name] = e;
                        order.Add(e.name);
                    }
            }
        }
    }
}
