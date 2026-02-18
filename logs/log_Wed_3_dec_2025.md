**Summary**

- Only **fwr1.sdp.mkat.karoo.kat.ac.za (port 1/24 on ing25g)** showed packet discards during 08:00–09:00 UTC on 2025‑12‑03. Prometheus’ `increase(switch_port_discard_packets_total…)` at 09:00 UTC reports **≈1 409 dropped packets (TX direction)**; RX discards stayed at 0. Neither **fwr2** (port 1/23) nor **fwr3** (port 1/22) recorded any discards in either direction.
- The discard burst lasted roughly **08:32–08:48 UTC**, with a smaller burst around **08:56–09:00 UTC**. Instantaneous drop rates peaked at **≈2.1 pps** (PromQL: `rate(switch_port_discard_packets_total{remote_name="fwr1…",direction="tx"}[5m])` sampled every 120 s).
- During the same periods, `rate(switch_port_bytes_total{…direction="tx"})` jumped from an idle baseline (~200 B/s) to **(0.8–2.7) Gb/s**. The port never hit line rate, which suggests **queue overruns caused by the receiver failing to drain bursts fast enough rather than link congestion**.
- Elasticsearch logs on **fwr1** align with the Prometheus spikes. At **08:29:47Z** and again at **08:54:27Z**, the **spead2/katsdpdatawriter** container logged repeated warnings:
  *“requested socket buffer size 619679744 but only received 268435456: refer to documentation for details on increasing buffer size”*.
  These warnings fall immediately before the two discard windows, pointing to **insufficient socket buffer space / kernel receive settings on fwr1** leading to back‑pressure and switch egress packet drops.
- `switch_port_error_packets_total` remained zero for all three nodes, ruling out physical-layer errors.

**Detailed findings**

| Node (remote) | Switch port | Direction with drops | Peak drop rate | Total drops 08:00–09:00 UTC | Traffic observation |
|---------------|------------|----------------------|----------------|------------------------------|---------------------|
| fwr1.sdp…     | 1/24       | TX (switch → host)   | ~2.1 pps @08:35 | ~1 409 packets (`increase` over 1 h) | Throughput spiked to 0.8–2.7 Gb/s in short bursts; RX path idle |
| fwr2.sdp…     | 1/23       | none                 | 0              | 0                            | Link traffic steady & low |
| fwr3.sdp…     | 1/22       | none                 | 0              | 0                            | Link traffic steady & low |

**Likely cause**

- The **spead2 data-writer on fwr1 cannot allocate the requested 620 MB socket buffer** (kernel limits it to 256 MB according to the warnings). When high-rate bursts arrive from ing25g, the reduced receive buffer fills quickly, the host stops acknowledging promptly, and the switch’s TX queue for port 1/24 overflows, producing the observed discards.
- No evidence of physical faults (CRC/errors) or similar behaviour on the sibling forwarders, so the issue is localized to **fwr1’s receive stack configuration**.

**Recommended actions**

1. **Increase UDP/TCP receive buffer limits on fwr1** (`net.core.rmem_max`, container `SO_RCVBUF`, etc.) so the spead2 writers can allocate the 619 MB they request. This should absorb the burstiness without forcing the switch to drop.
2. After adjusting buffers, **re-run Prometheus checks** (`rate`/`increase` queries above) to confirm the discard counters stop climbing during similar workloads.
3. Optionally **enable flow control or PFC** on port 1/24 if supported, to provide hardware back-pressure instead of drops.
4. Continue to monitor `switch_port_discard_packets_total` for fwr1; keep `switch_port_error_packets_total` in the watch list to ensure no physical issues emerge.

These steps should remove the bottleneck that is currently causing ing25g’s discards when sending data to fwr1, while fwr2 and fwr3 do not require changes at this time.

Elasticsearch shows numerous repeats of that exact warning. Querying `logstash-2025.12.03` for

```
"requested socket buffer size 619679744 but only received 268435456"
```

returned **48 hits**, all emitted by **host `fwr1.sdp.mkat.karoo.kat.ac.za`**, logger **`spead2`**, task `vis_writer.sdp_l0` (container image `katsdpdatawriter`). The first occurrence in this index is at **2025‑12‑03 08:29:47.558Z** and the latest at **10:11:33.193Z**, so the warnings continue well beyond the 08:00–09:00 window until the sysctl limits are raised.

These logs precisely match the buffer-shortage cause we discussed: the vis-writer keeps requesting a 619 MB `SO_RCVBUF` but is capped at 256 MB (`net.core.rmem_max`/`wmem_max`=268 435 456), leading to repeated WARNs and backpressure on ing25g port 1/24. Once you bump the sysctl values and reload, this log pattern should disappear; you can rerun the same Elasticsearch search afterward to confirm the warnings stop appearing.

### Details
Here are all 48 log hits that match
`"requested socket buffer size 619679744 but only received 268435456: refer to documentation for details on increasing buffer size"`
from `logstash-2025.12.03`, sorted by `@timestamp`:

| # | @timestamp (UTC) | Host | Task |
|---|------------------|------|------|
| 1 | 2025-12-03T08:29:47.558644Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
| 2 | 2025-12-03T08:29:47.559364Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
| 3 | 2025-12-03T08:29:47.559532Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
| 4 | 2025-12-03T08:29:47.559726Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
| 5 | 2025-12-03T08:38:51.025298Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
| 6 | 2025-12-03T08:38:51.025611Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
| 7 | 2025-12-03T08:38:51.479613Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
| 8 | 2025-12-03T08:38:51.480111Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
| 9 | 2025-12-03T08:54:26.970684Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|10 | 2025-12-03T08:54:26.971015Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|11 | 2025-12-03T08:54:27.329207Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|12 | 2025-12-03T08:54:27.329775Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|13 | 2025-12-03T09:05:21.274548Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|14 | 2025-12-03T09:05:21.274886Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|15 | 2025-12-03T09:05:21.719170Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|16 | 2025-12-03T09:05:21.719601Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|17 | 2025-12-03T09:12:48.648900Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|18 | 2025-12-03T09:12:48.649307Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|19 | 2025-12-03T09:12:48.998498Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|20 | 2025-12-03T09:12:48.998848Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|21 | 2025-12-03T09:19:24.068158Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|22 | 2025-12-03T09:19:24.068475Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|23 | 2025-12-03T09:19:24.522859Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|24 | 2025-12-03T09:19:24.523273Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|25 | 2025-12-03T09:26:11.156749Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|26 | 2025-12-03T09:26:11.157076Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|27 | 2025-12-03T09:26:11.501192Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|28 | 2025-12-03T09:26:11.501621Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|29 | 2025-12-03T09:37:28.084175Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|30 | 2025-12-03T09:37:28.084490Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|31 | 2025-12-03T09:37:28.535882Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|32 | 2025-12-03T09:37:28.536311Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|33 | 2025-12-03T09:48:03.044352Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|34 | 2025-12-03T09:48:03.044678Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|35 | 2025-12-03T09:48:03.044827Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|36 | 2025-12-03T09:48:03.400033Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|37 | 2025-12-03T09:53:37.129864Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|38 | 2025-12-03T09:53:37.130184Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|39 | 2025-12-03T09:53:37.577060Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|40 | 2025-12-03T09:53:37.577674Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|41 | 2025-12-03T10:01:56.285127Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|42 | 2025-12-03T10:01:56.285446Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|43 | 2025-12-03T10:01:56.635261Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|44 | 2025-12-03T10:01:56.635699Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|45 | 2025-12-03T10:11:32.736342Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|46 | 2025-12-03T10:11:32.736648Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|47 | 2025-12-03T10:11:33.192547Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |
|48 | 2025-12-03T10:11:33.192978Z | fwr1.sdp.mkat.karoo.kat.ac.za | vis_writer.sdp_l0 |

All entries come from the same host (`fwr1`) and process (`vis_writer.sdp_l0` via the `spead2` logger), which repeatedly failed to allocate the requested 619 MB socket buffer because `net.core.rmem_max`/`wmem_max` were limited to 268 MB. After you increase those sysctl limits and reload them, the warning should stop appearing; you can rerun the same Elasticsearch query to confirm it no longer returns fresh hits.


Here is a clean grouping of the timestamps into **natural “event bursts”** — i.e., records that occur within a few milliseconds of each other.
Each burst clearly corresponds to a **vis_writer.sdp_l0 flush/write cycle**.

---

# **Grouped Time Chunks**

## **Chunk 1 — 08:29:47.558–08:29:47.560**

Entries: **1–4**
These four events are tightly packed within **1.1 ms**.

---

## **Chunk 2 — 08:38:51.025–08:38:51.480**

Entries: **5–8**
Two sub-bursts here (~0.0003 s apart), then another ~0.45 s later, but still clearly one logical batch.

---

## **Chunk 3 — 08:54:26.970–08:54:27.330**

Entries: **9–12**
Two events at 26.97 s and two at 27.33 s — another compact cycle.

---

## **Chunk 4 — 09:05:21.274–09:05:21.720**

Entries: **13–16**
Two at 21.274 s and two at 21.719 s.

---

## **Chunk 5 — 09:12:48.649–09:12:48.999**

Entries: **17–20**
Two at 48.649 s and two at 48.999 s.

---

## **Chunk 6 — 09:19:24.068–09:19:24.523**

Entries: **21–24**
Two at 24.068 s and two at 24.523 s.

---

## **Chunk 7 — 09:26:11.157–09:26:11.501**

Entries: **25–28**
Two at 11.157 s and two at 11.501 s.

---

## **Chunk 8 — 09:37:28.084–09:37:28.536**

Entries: **29–32**
Two at 28.084 s and two at 28.535 s.

---

## **Chunk 9 — 09:48:03.044–09:48:03.400**

Entries: **33–36**
Three at 03.044 s and one ~0.36 s later.

---

## **Chunk 10 — 09:53:37.130–09:53:37.578**

Entries: **37–40**
Two at 37.130 s and two at 37.577 s.

---

## **Chunk 11 — 10:01:56.285–10:01:56.636**

Entries: **41–44**
Two at 56.285 s and two at 56.635 s.

---

## **Chunk 12 — 10:11:32.736–10:11:33.193**

Entries: **45–48**
Two at 32.736 s and two at 33.192 s.

---

# Summary Table

| Chunk | Time Window (UTC)   | Entries |
| ----- | ------------------- | ------- |
| 1     | 08:29:47.558–47.560 | 1–4     |
| 2     | 08:38:51.025–51.480 | 5–8     |
| 3     | 08:54:26.970–27.330 | 9–12    |
| 4     | 09:05:21.274–21.720 | 13–16   |
| 5     | 09:12:48.649–48.999 | 17–20   |
| 6     | 09:19:24.068–24.523 | 21–24   |
| 7     | 09:26:11.157–11.501 | 25–28   |
| 8     | 09:37:28.084–28.536 | 29–32   |
| 9     | 09:48:03.044–03.400 | 33–36   |
| 10    | 09:53:37.130–37.578 | 37–40   |
| 11    | 10:01:56.285–56.636 | 41–44   |
| 12    | 10:11:32.736–33.193 | 45–48   |

---

