
"""
real_event_store.py  —  Load real historical events and replay them
====================================================================
Replaces EventGenerator's Poisson process with actual FIRMS/GDACS events.

Key design:
  - Events are pre-loaded from combined_events.json
  - At each episode reset(seed), events are sub-selected based on the
    simulation epoch: events within ±30 days of the epoch are eligible
  - During simulation time-stepping, events "trigger" when sim_time_utc
    passes event's utc_start
  - Cloud cover is looked up from the ERA5 grid at (utc_start, lat, lon)
    instead of sampled from Beta distribution
"""
from __future__ import annotations
import json, os, datetime, math
from typing import List, Optional
import numpy as np

# Keep the same public interface as EventGenerator so callers don't change
from dynamic_event import DynamicEvent, EventType

_TYPE_MAP = {
    "wildfire":   EventType.WILDFIRE,
    "flood":      EventType.FLOOD,
    "earthquake": EventType.EARTHQUAKE,
    "eruption":   EventType.VOLCANIC_ERUPTION,
    "plume":      EventType.POLLUTION_PLUME,
}

_DURATION_S: dict[str, float] = {
    "wildfire":   12 * 3600,
    "flood":      72 * 3600,
    "earthquake":  3 * 3600,
    "eruption":   24 * 3600,
    "plume":       8 * 3600,
}


class RealEventStore:
    """
    Drop-in replacement for EventGenerator.
    
    Parameters
    ----------
    events_json : path to data/real_events/combined_events.json
    cloud_lookup : CloudLookup instance (ERA5 or MODIS based)
    epoch_utc   : simulation start UTC (matches TLE epoch)
    window_days : ±days around epoch to draw events from
    max_events  : cap on events per episode (prevents overload)
    rng         : numpy Generator for reproducibility
    """
    def __init__(
        self,
        events_json: str,
        cloud_lookup,           # CloudLookup instance (see Step 3)
        epoch_utc: datetime.datetime,
        window_days: int = 30,
        max_events:  int = 20,
        rng: Optional[np.random.Generator] = None,
    ):
        self._cloud  = cloud_lookup
        self._epoch  = epoch_utc
        self._window = window_days
        self._max    = max_events
        self._rng    = rng or np.random.default_rng(42)
        
        with open(events_json) as f:
            raw = json.load(f)
        self._all_events = raw
        self._active: List[DynamicEvent] = []
        self._pending: List[dict] = []   # raw records, not yet triggered
        self._triggered_ids: set = set()
        self._episode_events: List[dict] = []

    def reset(self, seed: Optional[int] = None):
        """Select events for the new episode. Call at env.reset()."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._active.clear()
        self._triggered_ids.clear()
        
        # Select events within the time window
        eligible = []
        lo = self._epoch - datetime.timedelta(days=self._window)
        hi = self._epoch + datetime.timedelta(days=self._window)
        for ev in self._all_events:
            try:
                t = _parse_utc(ev['utc_start'])
                if lo <= t <= hi:
                    eligible.append(ev)
            except Exception:
                continue
        
        # Sub-select up to max_events, weighted by priority
        if len(eligible) > self._max:
            weights = np.array([float(e.get('priority', 0.5)) for e in eligible])
            weights = weights / weights.sum()
            idxs = self._rng.choice(len(eligible), size=self._max, replace=False, p=weights)
            eligible = [eligible[i] for i in idxs]
        
        self._episode_events = eligible
        self._pending = list(eligible)  # will be consumed as sim time advances
        return len(eligible)

    def step(self, sim_utc: datetime.datetime) -> List[DynamicEvent]:
        """
        Trigger events whose utc_start <= sim_utc.
        Returns newly spawned DynamicEvent objects.
        Call at every env step.
        """
        spawned = []
        remaining = []
        for raw in self._pending:
            try:
                t_start = _parse_utc(raw['utc_start'])
            except Exception:
                continue
            if t_start <= sim_utc:
                ev = self._to_dynamic_event(raw, sim_utc)
                if ev is not None:
                    self._active.append(ev)
                    spawned.append(ev)
            else:
                remaining.append(raw)
        self._pending = remaining
        
        # Expire old events
        self._active = [e for e in self._active if not e.is_expired(sim_utc)]
        return spawned

    def get_active(self) -> List[DynamicEvent]:
        return list(self._active)

    # ── private ──────────────────────────────────────────────────────────────

    def _to_dynamic_event(
        self, raw: dict, now: datetime.datetime
    ) -> Optional[DynamicEvent]:
        ev_type = _TYPE_MAP.get(raw.get('type', 'wildfire'), EventType.WILDFIRE)
        lat = float(raw['lat'])
        lon = float(raw['lon'])
        
        # Real cloud cover from ERA5 at the event's time and location
        cloud = self._cloud.get(
            lat=lat, lon=lon,
            utc=_parse_utc(raw['utc_start'])
        )
        # For wildfires: add smoke bias (smoke increases apparent cloud coverage)
        if raw.get('type') == 'wildfire':
            frp = float(raw.get('frp_mw', 10.0))
            smoke_penalty = min(0.3, frp / 100.0)   # FRP 0–100 MW → 0–30% extra cloud
            cloud = min(1.0, cloud + smoke_penalty)
        
        duration_s = _DURATION_S.get(raw.get('type', 'wildfire'), 12 * 3600)
        
        try:
            return DynamicEvent(
                lat=lat,
                lon=lon,
                event_type=ev_type,
                priority=float(raw.get('priority', 0.5)),
                cloud_fraction=cloud,
                duration_s=duration_s,
                spawn_time=now,
            )
        except Exception as e:
            return None


def _parse_utc(s: str) -> datetime.datetime:
    """Parse ISO8601 or RFC2822 date strings robustly."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z",    # RFC2822 (GDACS RSS)
        "%a, %d %b %Y %H:%M:%S GMT",
    ):
        try:
            dt = datetime.datetime.strptime(s.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}")
