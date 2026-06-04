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

    def test_ratio_clamps_below_first_anchor(self):
        c = CapacityCurve(
            anchor_temps_c=(0.0, 10.0),
            anchor_ratios=(0.5, 1.0),
        )
        assert c.ratio(-50.0) == 0.5  # held at boundary, no extrapolation

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

    def test_boundary_clamping_below_cutoff(self):
        """Below the lowest anchor (AOU24 cutoff -26°C), we clamp at the
        boundary value rather than extrapolating to fantasy. Documented
        behavior — see module docstring for rationale and improvement path."""
        aou24 = get_profile("fujitsu_aou24rlxfwh")
        # At -40°C (well below cutoff), we hold at the cutoff value (0.54)
        # NOT extrapolate to a negative number.
        assert aou24.heat_curve.ratio(-40.0) == aou24.heat_curve.ratio(-26.0)
        assert aou24.heat_curve.ratio(-40.0) > 0  # not extrapolated to garbage

    def test_factor_dispatches_by_mode(self):
        p = get_profile("fujitsu_aou24rlxfwh")
        # Heating mode at -10°C should equal the heat curve's value there.
        assert p.factor(-10.0, "heat") == p.heat_curve.ratio(-10.0)
        # Cooling mode at 35°C (AHRI rated) should be 1.0.
        assert p.factor(35.0, "cool") == pytest.approx(1.0, abs=0.02)
