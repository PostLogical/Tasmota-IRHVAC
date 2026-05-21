"""Unit tests for the reference-governor / chatter-defense primitives.

Pure-logic coverage of QRefBiaser, ChatterTier, ChatterMonitor, and
ReferenceGovernor (the supervisor's building blocks).  The closed-loop
behaviour is exercised in the bench; these lock the per-method branch logic.
"""
import pytest

from custom_components.tasmota_irhvac.pi.ref_governor import (
    ChatterMonitor,
    ChatterTier,
    QRefBiaser,
    ReferenceGovernor,
)


# ── QRefBiaser ──────────────────────────────────────────────────────────


class TestQRefBiaser:
    def test_inactive_decays_toward_zero(self):
        b = QRefBiaser(decay_alpha=0.5)
        b._bias = 1.0
        out = b.update(raw_setpoint=28.4, active_and_in_db=False)
        assert out == 0.5  # 1.0 * (1 - 0.5)
        assert b.bias == 0.5

    def test_active_ema_tracks_quantization_residue(self):
        # q_error = round(28.4) - 28.4 = -0.4; target = -0.4 * gain
        b = QRefBiaser(gain=1.0, ema_alpha=1.0, max_bias_c=0.45)
        out = b.update(raw_setpoint=28.4, active_and_in_db=True)
        # ema_alpha=1 → bias jumps fully to clamped target (-0.4)
        assert out == pytest.approx(-0.4)

    def test_active_clamps_to_caps(self):
        b = QRefBiaser(gain=10.0, ema_alpha=1.0, max_bias_c=0.45)
        # q_error = round(28.4)-28.4 = -0.4; target = -4.0 → clamp to -0.45
        out = b.update(raw_setpoint=28.4, active_and_in_db=True)
        assert out == -0.45

    def test_directional_caps_override(self):
        b = QRefBiaser(gain=10.0, ema_alpha=1.0, max_bias_up_c=0.5, max_bias_down_c=0.2)
        # negative target clamps to -down=-0.2
        assert b.update(28.4, True) == -0.2
        b2 = QRefBiaser(gain=10.0, ema_alpha=1.0, max_bias_up_c=0.5, max_bias_down_c=0.2)
        # round(28.6)-28.6 = +0.4 → positive target clamps to +up=0.5
        assert b2.update(28.6, True) == 0.5


# ── ChatterTier ─────────────────────────────────────────────────────────


class TestChatterTier:
    def test_fires_at_threshold(self):
        t = ChatterTier(window_seconds=100, threshold=2, cooldown_seconds=10)
        assert t.tick(now_mono=0.0, new_event=True) is False  # 1 event
        assert t.tick(now_mono=1.0, new_event=True) is True   # 2 events ≥ threshold
        assert t.event_count == 2

    def test_cooldown_suppresses_after_fire(self):
        t = ChatterTier(window_seconds=100, threshold=2, cooldown_seconds=10)
        t.tick(0.0, True)
        assert t.tick(1.0, True) is True            # fires, cooldown until 11
        assert t.tick(5.0, True) is False           # within cooldown
        assert t.tick(20.0, True) is True           # cooldown expired, refires

    def test_stale_events_expire_outside_window(self):
        t = ChatterTier(window_seconds=10, threshold=2, cooldown_seconds=5)
        t.tick(0.0, True)
        # at t=15, the t=0 event is older than the 10s window → dropped
        assert t.tick(15.0, True) is False
        assert t.event_count == 1

    def test_no_new_event_just_expires(self):
        t = ChatterTier(window_seconds=10, threshold=1, cooldown_seconds=5)
        t.tick(0.0, True)              # fires (threshold 1)
        out = t.tick(20.0, False)      # no new event, t=0 expired
        assert out is False
        assert t.event_count == 0


# ── ChatterMonitor ──────────────────────────────────────────────────────


def _mono(window=100, threshold=2, cooldown=5):
    return ChatterMonitor(tiers=[ChatterTier(window, threshold, cooldown)])


class TestChatterMonitor:
    def test_first_tick_no_event(self):
        m = _mono()
        assert m.update(raw_setpoint=28.0, now_mono=0.0) is False
        assert m.tiers[0].event_count == 0  # no boundary cross on first sample

    def test_boundary_crossing_counts_event_and_fires(self):
        m = _mono(threshold=2)
        m.update(28.0, 0.0)                      # establishes last_rounded=28
        assert m.update(29.0, 1.0) is False      # 1 crossing (28→29)
        assert m.update(28.0, 2.0) is True       # 2nd crossing (29→28) → fires
        assert m.event_count == 2

    def test_no_crossing_no_event(self):
        m = _mono(threshold=1)
        m.update(28.1, 0.0)
        assert m.update(28.2, 1.0) is False      # round stays 28 → no event

    def test_max_window_and_sample_eviction(self):
        m = ChatterMonitor(tiers=[ChatterTier(60, 5, 5), ChatterTier(600, 5, 5)])
        assert m._max_window == 600
        m.update(28.0, 0.0)
        m.update(29.0, 700.0)                    # first sample older than 600 → evicted
        assert len(m._samples) == 1

    def test_duty_cycle_lean(self):
        m = _mono()
        # raw - round(raw): 28.3→+0.3 (above), 28.7→round 29→-0.3 (below)
        m.update(28.3, 0.0)
        m.update(28.7, 1.0)
        # 1 of 2 samples above → (1/2)*2 - 1 = 0.0
        assert m.duty_cycle_lean() == 0.0
        m.update(28.3, 2.0)                      # 2 of 3 above → (2/3)*2-1 = 1/3
        assert abs(m.duty_cycle_lean() - (2 / 3 * 2 - 1)) < 1e-9

    def test_duty_cycle_lean_empty(self):
        assert ChatterMonitor(tiers=[]).duty_cycle_lean() == 0.0

    def test_tier_diagnostics(self):
        m = ChatterMonitor(tiers=[ChatterTier(100, 2, 5), ChatterTier(100, 99, 5)])
        m.update(28.0, 0.0)
        m.update(29.0, 1.0)
        m.update(28.0, 2.0)                      # tier 0 fires (threshold 2)
        assert m.last_fired_tier == 0
        assert m.tier_event_counts == [2, 2]

    def test_event_count_empty_tiers(self):
        assert ChatterMonitor(tiers=[]).event_count == 0


# ── ReferenceGovernor ───────────────────────────────────────────────────


class TestReferenceGovernor:
    def test_normal_no_alarm_holds(self):
        g = ReferenceGovernor()
        v, d = g.step(r_user_c=20.0, room_c=21.0, chatter_alarm=False,
                      duty_lean=0.0, ki=0.15, now_mono=0.0)
        assert v == 20.0 and d == 0.0
        assert g.mode == "NORMAL" and not g.is_engaged

    def test_engage_direction_from_duty_lean(self):
        g = ReferenceGovernor(max_nudge_c=0.45)
        # duty_lean > 0 → commit to the under-represented (lower) side: -max
        v, d = g.step(20.0, 21.0, chatter_alarm=True, duty_lean=0.5,
                      ki=0.15, now_mono=0.0)
        assert g.nudge_c == -0.45 and g.mode == "NUDGE"
        assert v == 20.0 - 0.45
        assert d == 0.45 / 0.15            # bumpless: -new_nudge/ki = -(-0.45)/0.15

    def test_engage_room_sign_fallback_balanced_duty(self):
        g = ReferenceGovernor(max_nudge_c=0.45)
        # duty_lean ~ 0 → fall back to room sign; room>r_user → +max
        v, d = g.step(20.0, 22.0, chatter_alarm=True, duty_lean=0.0,
                      ki=0.15, now_mono=0.0)
        assert g.nudge_c == 0.45

    def test_no_engage_when_ki_zero(self):
        g = ReferenceGovernor()
        v, d = g.step(20.0, 21.0, chatter_alarm=True, duty_lean=0.5,
                      ki=0.0, now_mono=0.0)
        assert g.mode == "NORMAL" and g.nudge_c == 0.0

    def test_engage_kick_no_bumpless(self):
        g = ReferenceGovernor(engage_bumpless=False)
        v, d = g.step(20.0, 21.0, chatter_alarm=True, duty_lean=0.5,
                      ki=0.15, now_mono=0.0)
        assert g.mode == "NUDGE" and d == 0.0   # kick: integrator untouched

    def test_nudge_adaptive_relax_tracks_offset(self):
        g = ReferenceGovernor(max_nudge_c=0.45, adapt_rate=0.5)
        g.mode = "NUDGE"
        g.nudge_c = -0.45
        # target = clamp(room-r_user, ±max) = clamp(0.2) = 0.2; adj=(0.2-(-0.45))*0.5
        v, d = g.step(20.0, 20.2, chatter_alarm=False, duty_lean=0.0,
                      ki=0.15, now_mono=0.0)
        adj = (0.2 - (-0.45)) * 0.5
        assert abs(g.nudge_c - (-0.45 + adj)) < 1e-9
        assert abs(d - (-adj / 0.15)) < 1e-9    # relax bumpless

    def test_nudge_drift_release_after_duration(self):
        g = ReferenceGovernor(drift_band_c=0.5, drift_duration_sec=100.0, adapt_rate=0.0)
        g.mode = "NUDGE"
        g.nudge_c = 0.45
        # room far from r_user (|2.0| > band 0.5) → drift timer starts
        g.step(20.0, 22.0, chatter_alarm=False, duty_lean=0.0, ki=0.15, now_mono=0.0)
        assert g.mode == "NUDGE"                # not yet released
        # after drift_duration → release
        v, d = g.step(20.0, 22.0, chatter_alarm=False, duty_lean=0.0,
                      ki=0.15, now_mono=150.0)
        assert g.mode == "NORMAL" and g.nudge_c == 0.0
        assert d == 0.45 / 0.15                 # release bumpless: +nudge/ki

    def test_nudge_in_band_resets_drift_timer(self):
        g = ReferenceGovernor(drift_band_c=0.5, adapt_rate=0.0)
        g.mode = "NUDGE"
        g.nudge_c = 0.3
        g.step(20.0, 22.0, False, 0.0, 0.15, 0.0)     # out of band → timer set
        assert g._drift_started_mono is not None
        g.step(20.0, 20.1, False, 0.0, 0.15, 50.0)    # back in band → timer cleared
        assert g._drift_started_mono is None
