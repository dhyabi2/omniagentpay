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


# ---- order-bound verification: binding, one-to-one persistent claims, bounded lookups ----
import hashlib  # noqa: E402
import sqlite3  # noqa: E402

import nano_settlement as ns  # noqa: E402

_B32 = "13456789abcdefghijkmnopqrstuwxyz"
PAY_TO = "nano_1111111111111111111111111111111111111111111111111111hifc8npp"
H1 = "A" * 64
URLS = ["https://a.example/rpc", "https://b.example/rpc"]


def _enc(n, chars):
    out = ""
    for _ in range(chars):
        out = _B32[n & 31] + out
        n >>= 5
    return out


def _account(pub: bytes) -> str:
    chk = hashlib.blake2b(pub, digest_size=5).digest()[::-1]
    return "nano_" + _enc(int.from_bytes(pub, "big"), 52) + _enc(int.from_bytes(chk, "big"), 8)


def _block(order, rep_order_id=None, amount=None, bound=True):
    contents = {"link_as_account": order["pay_to"]}
    if bound:
        contents["representative"] = _account(
            ns.order_commitment(rep_order_id or order["order_id"], order["pay_to"], int(order["price_raw"])))
    return {"confirmed": "true", "subtype": "send", "contents": contents,
            "amount": str(amount if amount is not None else order["price_raw"])}


@pytest.fixture
def node(monkeypatch):
    state = {"block": None}

    async def fake(client, url, block_hash):
        return state["block"]

    monkeypatch.setattr(ns, "_block_info", fake)
    return state


def run(coro):
    return asyncio.run(coro)


def test_bound_block_pays_its_order_once(node):
    db = sqlite3.connect(":memory:")
    o = ns.issue_order(db, PAY_TO, 10**28)
    assert len(o["order_id"]) == 32
    node["block"] = _block(o)
    assert run(ns.verify_order_payment(db, o["order_id"], H1, URLS))["ok"] is True
    assert run(ns.verify_order_payment(db, o["order_id"], "B" * 64, URLS))["ok"] is False


def test_unbound_or_foreign_block_refused(node):
    db = sqlite3.connect(":memory:")
    a, b = ns.issue_order(db, PAY_TO, 10**28), ns.issue_order(db, PAY_TO, 10**28)
    node["block"] = _block(a, bound=False)  # right amount, right payee, no commitment
    assert run(ns.verify_order_payment(db, a["order_id"], H1, URLS))["reason"].endswith("block is not bound to this order")
    node["block"] = _block(a)  # committed to A
    assert run(ns.verify_order_payment(db, b["order_id"], H1, URLS))["ok"] is False
    assert run(ns.verify_order_payment(db, "nano_someone", H1, URLS))["reason"] == "unknown order"


def test_wrong_amount_and_expiry_refused(node):
    db = sqlite3.connect(":memory:")
    o = ns.issue_order(db, PAY_TO, 10**28, ttl_s=60)
    node["block"] = _block(o, amount=1)
    assert run(ns.verify_order_payment(db, o["order_id"], H1, URLS))["ok"] is False
    node["block"] = _block(o)
    late = run(ns.verify_order_payment(db, o["order_id"], H1, URLS, now=o["expires"] + 1))
    assert late["reason"] == "order expired"


def test_claim_survives_restart_and_is_one_to_one(node, tmp_path):
    path = str(tmp_path / "n.db")
    c1 = sqlite3.connect(path)
    a = ns.issue_order(c1, PAY_TO, 10**28)
    node["block"] = _block(a)
    assert run(ns.verify_order_payment(c1, a["order_id"], H1, URLS))["ok"]
    c1.close()
    c2 = sqlite3.connect(path)  # restart
    b = ns.issue_order(c2, PAY_TO, 10**28)
    node["block"] = _block(b)
    assert run(ns.verify_order_payment(c2, b["order_id"], H1, URLS)) == {
        "ok": False, "reason": "block already claimed by another order"}


def test_rpc_url_policy_and_response_cap():
    with pytest.raises(ValueError):
        ns.assert_safe_rpc_url("http://rpc.example.com")
    ns.assert_safe_rpc_url("https://rpc.nano.to")
    ns.assert_safe_rpc_url("http://127.0.0.1:7076")

    class R:
        content = b"x" * (ns.RPC_MAX_BYTES + 1)

        def raise_for_status(self):
            pass

    class C:
        async def post(self, *a, **k):
            return R()

    with pytest.raises(ValueError):
        run(ns._block_info(C(), "https://rpc.nano.to", H1))
    with pytest.raises(ValueError):
        run(ns._block_info(C(), "http://rpc.example.com", H1))
