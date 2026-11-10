# SpikeGLX census columns (`scripts/010_spikeglx_census.py`)

One row per probe folder (`raw_ephys_data/probeXX`) holding an `ap.meta`, in a session whose `raw_ephys_data` also holds
a `nidq.meta`. NP2.4 shanks (`probe00a`…`d`) are separate rows sharing one meta file. `<s>` is `nidq` or `probe`.
Times are in seconds, on each file's own clock unless stated.

## Identity

| column | definition |
|---|---|
| `lab`, `subject`, `date`, `number`, `probe` | from the folder path `<lab>/Subjects/<subject>/<date>/<number>/raw_ephys_data/<probe>` |
| `pid`, `eid`, `projects` | Alyx probe insertion matched on subject/date/number/probe name; empty when no insertion is registered |

## Files (full paths on popeye, `;`-joined when several match)

| column | definition |
|---|---|
| `<s>_meta_path` | `*.nidq.meta` / `*.ap.meta` |
| `<s>_raw_path` | `*.nidq.cbin|bin` / `*.ap.cbin|bin` |
| `<s>_wiring_path` | `*.nidq.wiring.json` / probe `*.wiring.json` |
| `<s>_sync_{times,channels,polarities}_path` | `_spikeglx_sync.*.npy` extracted by ibllib from the raw file |
| `<s>_n_meta_files` | number of matching meta files; > 1 means duplicates, the first (sorted) is used |
| `has_raw` | both raw binaries exist |

## Raw meta fields, verbatim strings from the file

`<s>.<key>` holds the value of `key` exactly as written in the `.meta` file (empty when absent). Every other non-table
key of every meta file is in `010_meta_raw.pqt` (path, key, value).

| key | streams | SpikeGLX meaning |
|---|---|---|
| `appVersion` | both | SpikeGLX version, yyyymmdd |
| `typeThis` | both | `nidq` or `imec` |
| `fileCreateTime` | both | wall-clock creation time of the file, 1 s resolution |
| `firstSample` | both | index of the file's first sample in the stream's own sample count since acquisition start |
| `fileTimeSecs` | both | file duration = samples in file / sampling rate |
| `fileSizeBytes` | both | size of the uncompressed binary |
| `nSavedChans` | both | channels per sample in the binary |
| `syncSourceIdx`, `syncSourcePeriod` | both | sync generator source and period (s) |
| `niSampRate` | nidq | calibrated nidq sampling rate (Hz) |
| `syncNiChan`, `syncNiChanType` | nidq | nidq line/type receiving the sync |
| `imSampRate` | probe | calibrated AP sampling rate (Hz) |
| `syncImInputSlot` | probe | slot of the module receiving the sync input |
| `imDatApi` | probe | imec API version |
| `imDatBs_fw`, `imDatBsc_fw`, `imDatBsc_hw` | probe | base-station firmware, base-station connect card firmware/hardware |
| `imDatHs_fw` | probe | headstage firmware |
| `imDatPrb_pn`, `imDatPrb_sn`, `imDatPrb_type` | probe | probe part number, serial, type code |
| `imDatPrb_slot`, `imDatPrb_port` | probe | base-station slot and port |

## Imec sync edges, per stream

Edges of the imec sync line only: channel from `wiring.json` (`SYNC_WIRING_DIGITAL == imec_sync`), else the ibllib 3B
default (nidq 3, probe 6). Edge times are seconds from the start of the file. The edges are cached on popeye in
`010_edges/<lab>_<subject>_<date>_<number>_<nidq|probeXX>.npz` (`times`, `polarities`).

| column | definition |
|---|---|
| `<s>_sync_status` | `ok`, or why edges are missing |
| `<s>_error` | read error, if any |
| `<s>_sync_channel`, `<s>_sync_channel_source` | channel used, `wiring` or `default` |
| `<s>_n_edges` | number of edges (rising + falling) |
| `<s>_pol_first`, `<s>_pol_last` | polarity of the first / last edge (+1 rising, −1 falling) |
| `<s>_t_first_edge_s`, `<s>_t_last_edge_s` | time of the first / last edge |
| `<s>_median_half_period_s` | median interval between consecutive edges (expected 0.5) |
| `<s>_max_gap_s` | longest interval between consecutive edges |
| `<s>_n_long_gaps` | intervals > 0.75 s: missing edges |
| `<s>_n_short_gaps` | intervals < 0.25 s: extra edges (bounces, glitches) |
| `<s>_n_same_polarity` | consecutive edges of equal polarity: an edge of the other polarity was lost |

## Derived

| column | definition |
|---|---|
| `<s>_t0_s` | `firstSample / SampRate`: file start on the stream's acquisition clock |
| `<s>_end_s` | `<s>_t0_s + fileTimeSecs` |
| `<s>_tail_s` | `fileTimeSecs − <s>_t_last_edge_s`; > 0.5 means the sync stopped before the file did |
| `start_lead_s` | `nidq_t0_s − probe_t0_s`; > 0: the probe file started first |
| `end_lead_s` | `nidq_end_s − probe_end_s`; > 0: the probe file ended first |
| `dT_s` | `probe.fileTimeSecs − nidq.fileTimeSecs` (= `start_lead_s − end_lead_s`, independent of `firstSample`) |
| `dn_edges` | `probe_n_edges − nidq_n_edges` |
| `firstSample_reliable` | both `appVersion` ≥ `20221012` |
| `truncated` | `abs(end_lead_s)` > 0.25 s when `firstSample_reliable`, else > 5 s |
| `truncated_stream` | stream that ended first when `truncated` |
