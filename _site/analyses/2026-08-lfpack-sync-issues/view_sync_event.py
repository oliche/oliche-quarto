"""
Stream and display raw AP band around a sync front-time anomaly, in viewephys: undestriped and
destriped, both with every active digital sync bit overlaid on the top axis (imec_sync in red,
every other -- unwired/floating on the probe file -- bit in gray). Reusable across sessions and
defect classes: edit the CONFIG block below (probe label -> pid, and the event time in each
probe's own local seconds) and rerun -- don't copy this file per case.

Needs viewephys on the path, e.g. run with:
    PYTHONPATH=<repo>/packages/viewephys/src python view_sync_event.py
"""
# %%
import json

import numpy as np
import pyqtgraph as pg
from brainbox.io.one import SpikeSortingLoader
from ibldsp.voltage import destripe
from neuropixel import trace_header
from one.api import ONE
from viewephys.viewer.qt import create_app

SYNC_COLOR = (220, 40, 40)  # imec_sync, the only bit actually wired on the probe file
NOISE_COLOR = (140, 140, 140)  # every other bit floats and just picks up chatter
BIT_OFFSET = 1.3  # vertical spacing between stacked digital traces in the header axis

app = create_app()

# ── CONFIG ────────────────────────────────────────────────────────────────────
# f2ea7211 duplicate_burst (probe): a 58.3ms short gap, a 33.3us micro-gap, then a 434.9ms gap,
# clustered within ~0.5s -- probe-local t, not nidq time (single probe, no nidq-time translation)
PIDS = {
    'probe00': 'f2ea7211-85f3-4394-b03e-1302a1dfe79c',
}
T_EVENT = 1955.46  # s, in each probe's own local time (see PIDS above for which)
PAD = 1.0  # s on each side
# ────────────────────────────────────────────────────────────────────────────

def stream_sync_digital(streamer, a, b):
    """
    Digital sync bits for samples [a, b) of an ap-band Streamer, decoded from the raw sync word
    of the same chunk `streamer[a:b]` would download -- Streamer has no public accessor for the
    digital trace, only for volts-converted analog channels, so this repeats its own chunk
    bookkeeping (see Streamer.read) to get a local, fully-downloaded spikeglx.Reader to call
    read_sync_digital on.

    Returns
    -------
    np.ndarray
        (b - a, 16) int8 array, one column per sync bit.
    """
    bounds = streamer.chunks['chunk_bounds']
    first_chunk = np.maximum(0, np.searchsorted(bounds, a) - 1)
    last_chunk = np.maximum(0, np.searchsorted(bounds, b) - 1)
    n0 = bounds[first_chunk]
    local_sr, _ = streamer._download_raw_partial(first_chunk=first_chunk, last_chunk=last_chunk)
    digital = local_sr.read_sync_digital(slice(a - n0, b - n0))
    local_sr.close()
    return digital


def get_imec_sync_bit(one, streamer):
    """
    Bit index of the 'imec_sync' line in the probe's own digital sync word, from its
    wiring.json -- every other bit is unwired on the probe file (only nidq carries the full
    camera/frame2ttl/audio digital map) and floats, so picking e.g. the lowest active bit
    instead picks up floating-input chatter rather than the real synchronisation pulse train.
    """
    wiring_file = one.load_dataset(streamer.eid, '*.wiring.json', collection=f'*{streamer.pname}',
                                    download_only=True)
    wiring = json.loads(wiring_file.read_text())
    for key, name in wiring['SYNC_WIRING_DIGITAL'].items():
        if name == 'imec_sync':
            return int(key.split('.')[1])
    raise ValueError(f'no imec_sync bit in {wiring_file}')


def plot_digital_bits(ev, t, digital, imec_sync_bit):
    """Stack every bit that toggles in this window in the header axis, offset so they don't
    overlap, imec_sync in red and every other (floating/unwired) bit in gray."""
    active_bits = np.where(digital.any(axis=0))[0]
    for rank, bit in enumerate(active_bits):
        color = SYNC_COLOR if bit == imec_sync_bit else NOISE_COLOR
        y = digital[:, bit].astype(float) + rank * BIT_OFFSET
        curve = pg.PlotCurveItem(x=t, y=y, connect='finite', pen=pg.mkPen(color=color, width=1))
        ev.plotItem_header_h.addItem(curve)
    # Y only -- plotItem_header_h.setXLink(plotItem_seismic) means autoRange() on this item would
    # also re-scale the shared X axis (zooming the whole window), not just this panel's Y.
    if active_bits.size:
        ev.plotItem_header_h.setYRange(-0.1, (active_bits.size - 1) * BIT_OFFSET + 1.1, padding=0)
    return active_bits


one = ONE()
viewers = {}
# shared per tag, not across tags -- raw and destriped sit on very different amplitude scales,
# so forcing one gain across both washes out whichever one it wasn't computed from. Both computed
# fresh via auto_gain() (not a hardcoded dB constant): the data is real volts (a_scalar=1 below),
# so a magic-number gain tuned for some other unit convention would silently stop meaning anything.
ref_gain = {'raw': None, 'destriped': None}

for probe, pid in PIDS.items():
    sr = SpikeSortingLoader(one=one, pid=pid).raw_electrophysiology(band='ap', stream=True)
    nc = sr.nc - 1  # drop sync channel
    a = int(round((T_EVENT - PAD) * sr.fs))
    b = int(round((T_EVENT + PAD) * sr.fs))
    raw = sr[a:b, :nc].T.astype(np.float32)  # (nc, ns), volts
    dst = destripe(raw, fs=sr.fs, neuropixel_version=sr.major_version)
    print(f"{probe} ({pid[:8]}): streamed {b - a} samples @ {sr.fs:.0f} Hz, "
          f"t=[{a / sr.fs:.3f}, {b / sr.fs:.3f}]s")

    digital = stream_sync_digital(sr, a, b)
    imec_sync_bit = get_imec_sync_bit(one, sr)
    t_sync = a / sr.fs + np.arange(digital.shape[0]) / sr.fs

    from viewephys.gui import viewephys
    channels = trace_header(version=sr.major_version)
    t0 = a / sr.fs
    for tag, data in [('raw', raw), ('destriped', dst)]:
        title = f"{probe} ({pid[:8]}) — {tag} AP, t={t0:.2f}-{b / sr.fs:.2f}s"
        # a_scalar=1: keep real volts on screen (the spikeglx.Reader/Streamer path above already
        # returns volts; viewephys's own default a_scalar=1e6 would silently redisplay in uV,
        # making it easy to misjudge saturation against the ADC's actual voltage range)
        ev = viewephys(data, sr.fs, channels=channels, title=title, t0=t0, a_scalar=1)
        if ref_gain[tag] is None:
            ref_gain[tag] = ev.ctrl.auto_gain()
        ev.ctrl.set_gain(ref_gain[tag])
        active_bits = plot_digital_bits(ev, t_sync, digital, imec_sync_bit)
        viewers[title] = ev
    print(f"{probe} ({pid[:8]}): imec_sync bit {imec_sync_bit}, active bits in window {active_bits.tolist()}")

print(f"\nOpened {len(viewers)} viewephys window(s) around t={T_EVENT}s.")
app.exec()
