"""
Tests for examples/nano_settlement.py — the Nano (XNO) settlement leg.

Run the offline part without any network:
    python -m pytest examples/test_nano_settlement.py -v

The live test hits two independent public Nano RPCs with a real confirmed send
and is skipped cleanly if the network is unavailable (never flakes the suite).
"""

import asyncio
import os
import sys
from decimal import Decimal

import pytest

# Allow importing nano_settlement.py (a sibling in the same examples/ dir)
sys.path.insert(0, os.path.dirname(__file__ or "."))

from nano_settlement import (
    DEMO_BLOCK_HASH,
    DEMO_PAYEE,
    usd_to_xno_raw,
    verify_nano_payment,
)


def test_usd_to_xno_raw_integer_and_never_underpays():
    """A USD invoice maps to an exact integer raw amount that is rounded UP
    (ceiling), so the seller is NEVER under-received on the 30-decimal Nano raw
    unit."""
    for price_usd, px in [
        (Decimal("0.001"), Decimal("0.55")),
        (Decimal("0.002"), Decimal("0.5")),
        (Decimal("0.001"), Decimal("0.30")),
        (Decimal("0.0001"), Decimal("0.55")),
    ]:
        raw = usd_to_xno_raw(price_usd, px)
        assert isinstance(raw, int)
        # sent XNO (raw/1e30) must be >= the exact XNO needed for the invoice
        xno_sent = Decimal(raw) / Decimal(10) ** 30
        xno_need = price_usd / px
        assert xno_sent >= xno_need, (price_usd, px, xno_sent, xno_need)


def test_usd_to_xno_raw_is_decimal_exact():
    """No float drift on a 30-decimal amount: 0.002 USD at 0.50 USD/XNO is
    exactly 4e27 raw."""
    assert usd_to_xno_raw(Decimal("0.002"), Decimal("0.5")) == 4 * (10**27)


def test_verify_fails_closed_with_one_endpoint():
    """One healthy RPC is NOT enough — a settlement claim needs >=2 independent
    nodes agreeing (fail closed, mirroring the repo's safety philosophy)."""
    result = asyncio.run(
        verify_nano_payment(
            block_hash=DEMO_BLOCK_HASH,
            expected_payee=DEMO_PAYEE,
            expected_amount_raw=205676479000000000000000000000000000000,
            rpc_urls=["https://rpc.nano.to"],  # only one endpoint
        )
    )
    assert result["ok"] is False
    assert ">=2 independent RPCs" in result["reason"]


def test_verify_fails_closed_when_amount_mismatch(monkeypatch):
    """A payment is NOT settled unless ALL queried nodes agree on the amount:
    node 2 sees a different amount, so ok=False even though node 1 confirms a
    send to the right payee (fail closed, not majority-wins)."""

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, **k):
            # Return the real confirmed shape on node 1, a DIFFERENT amount on node 2.
            if "rainstorm" in url:
                payload = "205676479000000000000000000000000000001"  # off by one raw
            else:
                payload = "205676479000000000000000000000000000000"
            node = {
                "block_account": DEMO_PAYEE,
                "confirmed": "true",
                "amount": payload,
                "contents": {"type": "send", "destination": DEMO_PAYEE},
            }
            return _FakeResponse(node)

    class _FakeResponse:
        def __init__(self, json):
            self._json = json

        def raise_for_status(self):
            pass

        def json(self):
            return self._json

    import nano_settlement as ns

    monkeypatch.setattr(ns.httpx, "AsyncClient", FakeClient)
    result = asyncio.run(
        verify_nano_payment(
            block_hash=DEMO_BLOCK_HASH,
            expected_payee=DEMO_PAYEE,
            expected_amount_raw=205676479000000000000000000000000000000,
        )
    )
    # Fail closed: ok is False; only the agreeing node is in confirmed_on.
    assert result["ok"] is False
    assert result["confirmed_on"] == ["https://rpc.nano.to"]


@pytest.mark.integration
def test_live_verification_two_independent_rpcs():
    """A real confirmed Nano send is verified on TWO independent public RPCs.
    Skipped if the network is down, so this never flakes the suite."""
    try:
        result = asyncio.run(
            verify_nano_payment(
                block_hash=DEMO_BLOCK_HASH,
                expected_payee=DEMO_PAYEE,
                expected_amount_raw=205676479000000000000000000000000000000,
            )
        )
    except Exception as exc:  # pragma: no cover - network flap
        pytest.skip(f"network unavailable: {exc}")
    # A node that could not be reached (DNS, proxy, 4xx/5xx) is an outage, not a
    # verdict on the block: skip rather than fail. A node that answered with a
    # mismatch ("send=..." reason) still fails the test.
    unreachable = [r for r in result["reasons"] if "send=" not in r]
    if unreachable:
        pytest.skip(f"public RPC unreachable: {unreachable}")
    assert result["ok"] is True, result["reasons"]
    assert len(result["confirmed_on"]) == 2
