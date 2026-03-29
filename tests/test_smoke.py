"""Smoke tests to verify test infrastructure works."""

import pytest
from custom_components.tasmota_irhvac.const import DOMAIN, PLATFORMS
from custom_components.tasmota_irhvac.pi_controller import PIController, _seed_buckets


def test_domain():
    """Verify domain constant."""
    assert DOMAIN == "tasmota_irhvac"


def test_platforms():
    """Verify platforms include climate and sensor."""
    assert "climate" in PLATFORMS
    assert "sensor" in PLATFORMS


def test_seed_buckets_heating():
    """Test feedforward bucket seeding for heating."""
    buckets = _seed_buckets(reference=15.0, slope=0.3)
    # At reference temp (15°C), offset should be 0
    assert buckets[15] == 0.0
    # Below reference, offset should be positive (heat more)
    assert buckets[0] > 0  # 15°C below reference → 0.3 * 15 = 4.5
    assert buckets[0] == pytest.approx(4.5)
    # Well above reference, offset should be 0
    assert buckets[18] == 0.0


def test_seed_buckets_cooling():
    """Test feedforward bucket seeding for cooling."""
    buckets = _seed_buckets(reference=25.0, slope=0.3, is_cooling=True)
    # At reference temp (25°C), offset should be 0
    assert buckets[24] == 0.0  # 24 is the closest bucket to 25
    # Above reference, offset should be negative (cool more)
    assert buckets[30] < 0
    assert buckets[30] == pytest.approx(-0.3 * 5)
    # Below reference, offset should be 0
    assert buckets[21] == 0.0
