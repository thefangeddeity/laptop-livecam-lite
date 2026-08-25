#!/usr/bin/env python3
"""
cv_notify.py -- notifications on track promotion, off the frame path.

Fires when the run-14 evidence accumulator promotes a track, never per
frame. A promotion is a real event: evidence has accumulated across
multiple detections past a threshold. A raw single-frame hit at 0.05 is
not, and notifying on one would mean an email every time a shadow moved.

Nothing here runs on the writer thread. The pipeline calls on_promotion(),
which snapshots the frame and enqueues; a worker thread does the encode and
the send. That ordering is deliberate and is the run-11 lesson: an SMTP
server that takes thirty seconds to answer must cost the feed nothing. The
queue is bounded and drops rather than blocks, because a backlog of stale
notifications is worth less than a live stream.

Apprise rather than smtplib: the backend is a URL. mailto:// today, and a
push service or webhook later is a config change with no code change. That
is the whole reason for the dependency, so nothing in here may assume the
destination is email.
"""

import os
import queue
import threading
import time

try:
    import apprise as _apprise
except Exception:      # optional dependency; absence disables notification
    _apprise = None

import cv2


class Notifier:
    """Queue-and-forget notifier keyed on track promotion.

    Construct once and keep. send() never blocks the caller for longer than
    a queue put on a bounded queue, which is O(1) and never waits because
    the queue is non-blocking.
    """

    def __init__(self, urls=None, classes=('human',), cooldown_sec=300,
                 enabled=False, snapshot_dir=None, queue_size=8,
                 log=None):
        self.enabled = bool(enabled) and _apprise is not None and bool(urls)
        self.urls = [u.strip() for u in (urls or []) if u.strip()]
        self.classes = {c.strip().lower() for c in classes if c.strip()}
        self.cooldown_sec = float(cooldown_sec)
        self.snapshot_dir = snapshot_dir or '/tmp'
        self._log = log or (lambda msg: None)

        self._q = queue.Queue(maxsize=int(queue_size))
        self._last_sent = {}        # class -> monotonic timestamp
        self._notified_tracks = set()
        self._thread = None
        self._stopping = threading.Event()

        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            self._log(f"notify: enabled, {len(self.urls)} destination(s), "
                      f"classes={sorted(self.classes)}, "
                      f"cooldown={self.cooldown_sec:.0f}s")
        elif urls and _apprise is None:
            self._log("notify: apprise not installed; notifications disabled")

    # ── called from the frame path; must stay cheap ─────────────────────
    def on_promotion(self, track, frame):
        """A track just crossed the promotion threshold.

        Returns True if a notification was enqueued. Snapshots by reference
        only -- the copy happens here because the caller reuses its buffer,
        but encoding and sending do not.
        """
        if not self.enabled:
            return False
        cls = (track.cls or '').lower()
        if cls not in self.classes:
            return False
        # One notification per track, ever. A track that is promoted, decays
        # below threshold and re-promotes is the same animal in the same
        # visit, not a second event.
        if track.track_id in self._notified_tracks:
            return False

        now = time.monotonic()
        last = self._last_sent.get(cls)
        if last is not None and (now - last) < self.cooldown_sec:
            # Cooldown is per class, and is what stops a cat that settles in
            # and gets repeatedly re-promoted from sending forty emails.
            self._notified_tracks.add(track.track_id)
            return False

        try:
            self._q.put_nowait({
                'cls': cls,
                'track_id': track.track_id,
                'confidence': float(track.confidence),
                'evidence': float(getattr(track, 'evidence', 0.0)),
                'when': time.time(),
                'frame': frame.copy(),
            })
        except queue.Full:
            # Dropping is correct: the frame path must never wait on the
            # notifier, and a queued backlog of old sightings has no value.
            self._log("notify: queue full, dropped a notification")
            return False

        self._last_sent[cls] = now
        self._notified_tracks.add(track.track_id)
        return True

    def forget_track(self, track_id):
        self._notified_tracks.discard(track_id)

    def close(self):
        self._stopping.set()

    # ── worker thread ───────────────────────────────────────────────────
    def _worker(self):
        while not self._stopping.is_set():
            try:
                item = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._send(item)
            except Exception as exc:
                # Never propagate: a failed send is a log line, not an
                # outage. The stream does not depend on this succeeding.
                self._log(f"notify: send failed: {exc}")
            finally:
                self._q.task_done()

    def _send(self, item):
        path = os.path.join(
            self.snapshot_dir,
            f"livecam-{item['cls']}-{int(item['when'])}-{item['track_id']}.jpg")
        wrote = False
        try:
            # The snapshot is the original feed, not the sharpie or CV
            # render. This is evidence for a person to look at, not input to
            # a machine, and a line drawing of a cat proves nothing.
            bgr = cv2.cvtColor(item['frame'], cv2.COLOR_RGB2BGR)
            wrote = cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            ap = _apprise.Apprise()
            for u in self.urls:
                ap.add(u)

            stamp = time.strftime('%Y-%m-%d %H:%M:%S',
                                  time.localtime(item['when']))
            title = f"Livecam: {item['cls']} detected"
            body = (f"{item['cls']} detected at {stamp}\n"
                    f"track {item['track_id']}, "
                    f"confidence {item['confidence']*100:.0f}%, "
                    f"evidence {item['evidence']:.2f}")

            ok = ap.notify(title=title, body=body,
                           attach=path if wrote else None)
            self._log(f"notify: {item['cls']} track {item['track_id']} "
                      f"{'sent' if ok else 'FAILED'}")
        finally:
            # Snapshots are transient by design -- recording is a separate
            # concern with its own retention budget. Clean up either way.
            if wrote:
                try:
                    os.unlink(path)
                except Exception:
                    pass


def make_notifier(denv, log=None):
    """Build from device.env, or a disabled Notifier if not configured."""
    denv = denv or {}

    def _get(key, default):
        v = denv.get(key)
        return default if v is None or str(v).strip() == '' else str(v).strip()

    enabled = _get('NOTIFY_ENABLED', '0') not in ('0', 'false', 'no', 'off')
    urls = [u for u in _get('NOTIFY_APPRISE_URLS', '').split(',') if u.strip()]
    classes = [c for c in _get('NOTIFY_CLASSES', 'human').split(',') if c.strip()]
    try:
        cooldown = float(_get('NOTIFY_COOLDOWN_SEC', '300'))
    except ValueError:
        cooldown = 300.0

    return Notifier(urls=urls, classes=classes, cooldown_sec=cooldown,
                    enabled=enabled, log=log)
