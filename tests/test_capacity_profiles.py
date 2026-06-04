"""Unit tests for HP capacity profiles.

The profile shapes themselves are calibrated against published data (NEEP
cold-climate ASHP product list, ASHRAE 90.1, AHRI ratings), so changing the
ratio values would invalidate the lit grounding. These tests pin the contract
(monotonicity, anchor behavior, rating-point normalization), not the specific
numbers — that lets the lit-grounded values evolve as better data sources
become available.
"""

import pytest

from custom_components.tasmota_irhvac.pi.capacity_profiles import (
    AHRI_COOL_RATING_C,
    AHRI_HEAT_RATING_C,
    CapacityCurve,
    CapacityProfile,
    available_profile_names,
    get_profile,
)


class TestCapacityCurve:
    def test_validates_anchor_lengths_match(self):
        with pytest.raises(ValueError, match="equal length"):
            CapacityCurve(anchor_temps_c=(0.0, 10.0), anchor_ratios=(1.0,))

    def test_validates_at_least_two_anchors(self):
        with pytest.raises(ValueError, match="at least 2"):
            CapacityCurve(anchor_temps_c=(0.0,), anchor_ratios=(1.0,))

    def test_validates_strictly_ascending_temps(self):
        with pytest.raises(ValueError, match="ascending"):
            CapacityCurve(
                anchor_temps_c=(0.0, 10.0, 5.0),
                anchor_ratios=(0.5, 1.0, 0.8),
            )

    def test_validates_non_negative_ratios(self):
        """Ratios must be ≥ 0 (zero allowed for 'HP off below cutoff' semantics
        used by legacy fixed-speed and cliff-cutoff profiles). Negative ratios
        are rejected as nonphysical."""
        # Zero is allowed — represents HP-off below cutoff temperature.
        c = CapacityCurve(
            anchor_temps_c=(-30.0, 0.0, 10.0),
            anchor_ratios=(  0.0,  0.5, 1.0),
        )
        assert c.ratio(-30.0) == 0.0
        # Negative is rejected — nonphysical.
        with pytest.raises(ValueError, match=">= 0"):
            CapacityCurve(
                anchor_temps_c=(0.0, 10.0),
                anchor_ratios=(-0.1, 1.0),
            )

    def test_ratio_returns_anchor_at_exact_temp(self):
        c = CapacityCurve(
            anchor_temps_c=(-10.0, 0.0, 10.0),
            anchor_ratios=( 0.5,   0.8, 1.0),
        )
        assert c.ratio(-10.0) == 0.5
        assert c.ratio(0.0) == 0.8
        assert c.ratio(10.0) == 1.0

    def test_ratio_linear_interpolation_midpoint(self):
        c = CapacityCurve(
            anchor_temps_c=(0.0, 10.0),
            anchor_ratios=(0.5, 1.5),
        )
        assert c.ratio(5.0) == pytest.approx(1.0)

    def test_ratio_clamps_below_first_anchor_when_no_cutoff_set(self):
        """When extrapolation_cutoff_c is None (legacy default), the ratio
        below the lowest anchor clamps to the boundary value. This preserves
        backward-compatible behavior for curves that don't specify a cutoff."""
        c = CapacityCurve(
            anchor_temps_c=(0.0, 10.0),
            anchor_ratios=(0.5, 1.0),
        )
        assert c.ratio(-50.0) == 0.5  # held at boundary, no extrapolation

    def test_ratio_linearly_extrapolates_below_first_anchor_when_cutoff_set(self):
        """With extrapolation_cutoff_c set, capacity drops linearly from the
        boundary ratio at the lowest anchor to 0 at the cutoff, and stays at
        0 below the cutoff. See module docstring for the 3× slope derivation
        rule used to pick cutoffs for the built-in profiles."""
        c = CapacityCurve(
            anchor_temps_c=(0.0, 10.0),
            anchor_ratios=(0.5, 1.0),
            extrapolation_cutoff_c=-20.0,
        )
        # At cutoff: 0
        assert c.ratio(-20.0) == 0.0
        # Below cutoff: still 0
        assert c.ratio(-50.0) == 0.0
        # Midway between cutoff and lowest anchor: half the boundary ratio.
        assert c.ratio(-10.0) == pytest.approx(0.25)
        # Just above cutoff: small positive value.
        assert c.ratio(-19.0) == pytest.approx(0.025, abs=0.005)
        # Just below the lowest anchor: close to boundary ratio.
        assert c.ratio(-1.0) == pytest.approx(0.475, abs=0.005)

    def test_extrapolation_cutoff_must_be_below_lowest_anchor(self):
        """Cutoff at or above the lowest anchor is rejected — extrapolation
        below the anchor range needs a lower bound to drop to zero at."""
        with pytest.raises(ValueError, match="below the lowest anchor"):
            CapacityCurve(
                anchor_temps_c=(-10.0, 0.0),
                anchor_ratios=(0.5, 1.0),
                extrapolation_cutoff_c=-10.0,  # equal to lowest anchor
            )
        with pytest.raises(ValueError, match="below the lowest anchor"):
            CapacityCurve(
                anchor_temps_c=(-10.0, 0.0),
                anchor_ratios=(0.5, 1.0),
                extrapolation_cutoff_c=-5.0,  # above the lowest anchor
            )

    def test_ratio_clamps_above_last_anchor(self):
        c = CapacityCurve(
            anchor_temps_c=(0.0, 10.0),
            anchor_ratios=(0.5, 1.0),
        )
        assert c.ratio(50.0) == 1.0


class TestBuiltinProfileContracts:
    """Profile-shape contracts grounded in HP physics, not specific numbers."""

    def test_unknown_profile_is_identity(self):
        """The "unknown" profile should produce ratio=1.0 everywhere (matches
        legacy constant-k_c behavior). Critical for backwards compatibility."""
        p = get_profile("unknown")
        for t in (-30.0, -10.0, 0.0, 8.33, 20.0, 35.0, 50.0):
            assert p.heat_curve.ratio(t) == 1.0
            assert p.cool_curve.ratio(t) == 1.0

    def test_unknown_name_falls_back_to_unknown_profile(self):
        p = get_profile("not_a_real_profile_name")
        assert p.name == "unknown"

    @pytest.mark.parametrize("profile_name", [
        "standard_inverter", "cold_climate_inverter", "fixed_speed",
    ])
    def test_heat_curve_normalized_at_rating_point(self, profile_name):
        """All heating curves should have ratio = 1.0 at AHRI 47°F rating point.
        This is the calibration baseline; ``k_c_rated`` we identify is the
        47°F-equivalent gain."""
        p = get_profile(profile_name)
        assert p.heat_curve.ratio(AHRI_HEAT_RATING_C) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("profile_name", [
        "standard_inverter", "cold_climate_inverter", "fixed_speed",
    ])
    def test_cool_curve_normalized_at_rating_point(self, profile_name):
        """All cooling curves should have ratio = 1.0 at AHRI 95°F rating point."""
        p = get_profile(profile_name)
        assert p.cool_curve.ratio(AHRI_COOL_RATING_C) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("profile_name", [
        "standard_inverter", "cold_climate_inverter", "fixed_speed",
    ])
    def test_heat_capacity_drops_below_rating_point(self, profile_name):
        """Below 47°F, all real heating curves should show capacity loss
        (ratio < 1.0). The cold-climate curve loses less, but still drops."""
        p = get_profile(profile_name)
        assert p.heat_curve.ratio(-10.0) < 1.0
        assert p.heat_curve.ratio(0.0) < 1.0

    def test_cold_climate_retains_more_capacity_than_standard_at_low_temp(self):
        """The defining feature of cold-climate HPs is better cold-weather
        capacity retention. Standard inverter at -15°C should be substantially
        below cold-climate at the same temperature."""
        std = get_profile("standard_inverter")
        cc = get_profile("cold_climate_inverter")
        # At -15°C (~5°F), cold-climate should retain at least 25% more capacity
        # than standard.
        ratio_std = std.heat_curve.ratio(-15.0)
        ratio_cc = cc.heat_curve.ratio(-15.0)
        assert ratio_cc > ratio_std + 0.20, (
            f"cold-climate {ratio_cc:.2f} vs standard {ratio_std:.2f}"
        )

    def test_fixed_speed_loses_more_capacity_than_inverter_in_cold(self):
        """Fixed-speed compressors don't have variable-speed boost; cold-
        weather capacity loss is sharper than inverter HPs."""
        fixed = get_profile("fixed_speed")
        std = get_profile("standard_inverter")
        # At -10°C (~14°F), fixed-speed should lose MORE capacity than standard
        # inverter (worse retention).
        ratio_fixed = fixed.heat_curve.ratio(-10.0)
        ratio_std = std.heat_curve.ratio(-10.0)
        assert ratio_fixed < ratio_std, (
            f"fixed_speed {ratio_fixed:.2f} ≥ standard {ratio_std:.2f}"
        )

    def test_class_ordering_at_extreme_cold(self):
        """HP classes must follow a consistent physical ordering of cold-
        weather capability. Fixed-speed fails first, then standard inverter,
        then Fujitsu hyperheat (≈ NEEP CC v4.0), then generic cold-climate
        last. Prevents future drift in the per-class extrapolation cutoffs.

        Stage 1e (2026-06-03) — derived from the 3× lowest-segment-slope
        rule on each curve's published anchors. See module docstring."""
        # Spot-check at -28°C (where fixed_speed and standard_inverter have
        # already cut off, but Fujitsu/CC are still delivering).
        fixed = get_profile("fixed_speed").heat_curve
        std = get_profile("standard_inverter").heat_curve
        fuj24 = get_profile("fujitsu_aou24rlxfwh").heat_curve
        cc = get_profile("cold_climate_inverter").heat_curve
        # At -28°C: standard's cutoff (≈ -28°C), so it should be near 0;
        # fixed-speed cut off long ago at -17°C; Fujitsu and CC still active.
        assert fixed.ratio(-28.0) == 0.0
        assert std.ratio(-28.0) == pytest.approx(0.0, abs=0.05)
        assert fuj24.ratio(-28.0) > 0.30  # well above any cutoff
        assert cc.ratio(-28.0) > 0.35
        # At -38°C: only generic cold-climate still delivering meaningful
        # capacity; Fujitsu (cutoff -38°C) at 0, others long gone.
        assert std.ratio(-38.0) == 0.0
        assert fuj24.ratio(-38.0) == 0.0
        assert cc.ratio(-38.0) > 0.10
        # Verify monotonic class ordering at a mid-cold temp (-15°C).
        # Each class should be no worse than the next-fragile class.
        r_fixed = fixed.ratio(-15.0)
        r_std = std.ratio(-15.0)
        r_fuj24 = fuj24.ratio(-15.0)
        r_cc = cc.ratio(-15.0)
        assert r_fixed <= r_std, f"fixed {r_fixed} > std {r_std}"
        assert r_std <= r_fuj24, f"std {r_std} > fuj24 {r_fuj24}"
        # Fujitsu and generic CC should be similar (Fujitsu slightly worse).
        assert r_fuj24 <= r_cc + 0.05, f"fuj24 {r_fuj24} > cc {r_cc} + 0.05"

    @pytest.mark.parametrize("profile_name", [
        "standard_inverter", "cold_climate_inverter", "fixed_speed",
    ])
    def test_heat_curve_is_monotonic_increasing(self, profile_name):
        """Heating capacity should monotonically increase with outdoor temp
        across the operational range (lit-typical residential HPs)."""
        p = get_profile(profile_name)
        temps = sorted(p.heat_curve.anchor_temps_c)
        ratios = [p.heat_curve.ratio(t) for t in temps]
        for i in range(1, len(ratios)):
            assert ratios[i] >= ratios[i - 1], (
                f"{profile_name} heat curve non-monotonic at {temps[i]}°C"
            )


class TestProfileRegistry:
    def test_available_profile_names_includes_all_documented(self):
        names = available_profile_names()
        assert "unknown" in names
        assert "standard_inverter" in names
        assert "cold_climate_inverter" in names
        assert "fixed_speed" in names
        assert "fujitsu_aou24rlxfwh" in names
        assert "fujitsu_aou36rlxfzh" in names

    def test_each_named_profile_resolves_to_distinct_instance(self):
        """No two named profiles should have the same name (otherwise we'd
        get confusing behavior on profile selection)."""
        profiles = [get_profile(n) for n in available_profile_names()]
        names = [p.name for p in profiles]
        assert len(set(names)) == len(names)


class TestFujitsuSpecData:
    """Pin the profiles for the user-installed Fujitsu equipment against the
    extracted Design & Technical Manual data points (source comments in
    capacity_profiles.py and tests/hvac_bench/house_profiles.py).

    These tests catch any drift between the production capacity_profiles and
    the bench's house_profiles (FUJITSU_HYPERHEAT_CAPACITY,
    FUJITSU_AOU36RLXFZH_CAPACITY) so the bench thermal model and the
    production fit see the same HP physics.

    Tolerance: 0.02 absolute on ratio — manufacturer specs are rounded to
    two decimals and our piecewise-linear is exact at anchors.
    """

    def _check(self, profile_name, expected_points):
        p = get_profile(profile_name)
        for t, expected_ratio in expected_points:
            actual = p.heat_curve.ratio(t)
            assert abs(actual - expected_ratio) <= 0.02, (
                f"{profile_name} at {t}°C: expected {expected_ratio:.2f}, "
                f"got {actual:.4f}"
            )

    def test_fujitsu_aou24rlxfwh_matches_spec_anchors(self):
        # Fujitsu 24RLXFW1 Design & Technical Manual heating capacity table,
        # ASU24RLF head at 70°F indoor (pdf-extracted 2026-05-17).
        self._check("fujitsu_aou24rlxfwh", [
            (-26.0, 0.54),  # cutoff
            (-20.6, 0.62),
            (-15.0, 0.70),
            (-10.0, 0.74),
            ( -5.0, 0.81),
            (  0.0, 0.89),
            (  5.0, 0.97),
            (  8.33, 1.00),  # AHRI rated
            ( 10.0, 1.02),
            ( 15.0, 0.97),
        ])

    def test_fujitsu_aou36rlxfzh_matches_spec_anchors(self):
        # AOU36RLXFZH Design & Technical Manual heating capacity table,
        # 36 kBtu connecting-capacity row, 70°F indoor (pdf-extracted 2026-05-17).
        self._check("fujitsu_aou36rlxfzh", [
            (-26.0, 0.53),  # cutoff
            (-20.6, 0.60),  # cliff edge
            (-15.0, 0.87),  # above cliff
            (-10.0, 0.92),
            ( -5.0, 0.97),
            (  0.0, 1.00),  # flat top starts
            (  5.0, 1.00),
            (  8.33, 1.00),  # AHRI rated
            ( 15.0, 1.00),
        ])

    def test_fujitsu_aou24_retains_more_capacity_than_standard_inverter(self):
        """The whole point of cold-climate Hyper-Heat units: capacity retention
        at low outdoor temps. AOU24 at -15°C should retain substantially more
        than the generic standard_inverter."""
        aou24 = get_profile("fujitsu_aou24rlxfwh").heat_curve.ratio(-15.0)
        std = get_profile("standard_inverter").heat_curve.ratio(-15.0)
        assert aou24 > std + 0.20, (
            f"AOU24 {aou24:.2f} vs standard {std:.2f} — insufficient cold-weather advantage"
        )

    def test_fujitsu_aou36_has_wide_flat_top(self):
        """AOU36 has factor ~1.0 from 0°C up through 15°C+ — the multi-split's
        capacity is governed by the indoor heads in the moderate range, so
        outdoor temp doesn't matter much there."""
        p = get_profile("fujitsu_aou36rlxfzh")
        for t in (0.0, 5.0, 10.0, 15.0):
            assert abs(p.heat_curve.ratio(t) - 1.0) <= 0.02, (
                f"AOU36 at {t}°C should be ≈1.0; got {p.heat_curve.ratio(t):.3f}"
            )

    def test_sub_anchor_linear_extrapolation_to_zero(self):
        """Below the lowest documented anchor (AOU24 at -26°C, 54%), capacity
        drops LINEARLY to 0 at the configured extrapolation_cutoff_c (-38°C
        for AOU24 per the 3× slope rule).

        Stage 1e (2026-06-03) replaced the older "boundary clamp" policy with
        a defensible linear extrapolation: the boundary clamp at 54% was no
        more honest than a cliff to 0%; both were made-up extrapolations.
        Linear-to-zero at a physically-grounded cutoff is more consistent
        with actual HP class capability (cold-climate models keep running
        further than standard inverters, but eventually all degrade to 0)."""
        aou24 = get_profile("fujitsu_aou24rlxfwh")
        # Cutoff is at -38°C; capacity is 0 at and below.
        assert aou24.heat_curve.ratio(-38.0) == 0.0
        assert aou24.heat_curve.ratio(-50.0) == 0.0
        # Midway between cutoff (-38°C, 0%) and the lowest anchor
        # (-26°C, 54%): should be 27% — half of boundary ratio.
        midpoint = (-38.0 + -26.0) / 2  # -32°C
        assert aou24.heat_curve.ratio(midpoint) == pytest.approx(0.27, abs=0.005)
        # Just above cutoff: still positive, dropping toward 0.
        assert 0.0 < aou24.heat_curve.ratio(-37.0) < 0.10
        # Just below lowest anchor: close to boundary ratio.
        assert aou24.heat_curve.ratio(-27.0) == pytest.approx(0.495, abs=0.01)

    def test_factor_dispatches_by_mode(self):
        p = get_profile("fujitsu_aou24rlxfwh")
        # Heating mode at -10°C should equal the heat curve's value there.
        assert p.factor(-10.0, "heat") == p.heat_curve.ratio(-10.0)
        # Cooling mode at 35°C (AHRI rated) should be 1.0.
        assert p.factor(35.0, "cool") == pytest.approx(1.0, abs=0.02)
