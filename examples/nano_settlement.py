"""
Nano (XNO) settlement leg for OmniAgentPay
==========================================

A runnable, keyless example that shows how an OmniAgentPay agent can settle a
machine-to-machine payment on the Nano rail (network ``nano:mainnet``) beside the
existing USDC transfer / x402 / CCTP adapters.

Why a Nano leg?
    OmniAgentPay routes each ``pay()`` through a protocol adapter (transfer, x402,
    gateway). Every one of those today settles a fee (USDC gas / CCTP bridging /
    x402 facilitator overhead). Nano settles in *zero* fee and confirms in ~1s,
    peer-to-peer, with no issuer that can freeze the funds. For a sub-USD per-call
    micropayment that is the difference between the rail being economic or not.

Keyless verification
    Unlike the USDC adapters (which need a Circle API key and a developer
    wallet), a Nano payment is verified by READING the open ledger: call
    ``block_info`` on *two independent public Nano RPCs* and check the block is a
    confirmed ``send`` to the expected payee for the expected amount. No API key,
    no card, no wallet bootstrap. Fail closed: one healthy endpoint is not
    enough; both must agree.

What this example does
    1. Fetches a live XNO price so a USD invoice maps to the exact raw amount
       (Nano has 30 decimals, raw = 1e-30 XNO).
    2. Builds the payment requirements an x402 ``network: nano:mainnet`` response
       would carry, mirroring how ``PaymentRequirements.from_response`` already
       parses ``scheme`` / ``network`` / ``amount`` / ``recipient``.
    3. Verifies a Nano send on-chain across two independent public RPCs,
       fail-closed, using a real confirmed send as the live sample.

Run it (no keys needed):
    python examples/nano_settlement.py

The example is written to mirror the repo's existing ``examples/x402_client_demo.py``
style and needs only ``httpx`` (already a project dependency).
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import sqlite3
import time
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from urllib.parse import urlparse

import httpx

# Two independent public Nano RPCs. A settlement claim is only trusted when both
# agree (each node keeps its own copy of the ledger; no single node is the source
# of truth). Wrap calls in try/except so a transient outage never blocks the run.
DEFAULT_RPC_URLS = [
    "https://rpc.nano.to",
    "https://rainstorm.city/api",
]

# A real confirmed send on the Nano mainnet (height 42 of nano_3t6k3…), verified
# live against both RPCs above. Used as the sample block for the demo run.
DEMO_BLOCK_HASH = "ECCB8CB65CD3106EDA8CE9AA893FEAD497A91BCA903890CBD7A5C59F06AB9113"
DEMO_PAYER = "nano_3t6k35gi95xu6tergt6p69ck76ogmitsa8mnijtpxm9fkcm736xtoncuohr3"
DEMO_PAYEE = "nano_1111111111111111111111111111111111111111111111111111hifc8npp"
DEMO_AMOUNT_RAW = "205676479000000000000000000000000000000"  # 0.000205676479 XNO

# Nano has 30 decimal places. raw = XNO * 10**30 is the smallest unit, exactly as
# USDC uses 6 decimals for its smallest unit (see get_amount_usdc in protocols/x402.py).
XNO_DECIMALS = 30
RAW_PER_XNO = Decimal(10) ** XNO_DECIMALS


@dataclass
class NanoRequirements:
    """A parsed ``nano:mainnet`` settlement requirement, mirroring the fields
    OmniAgentPay already parses from an x402 402 response (paymentAddress /
    recipient, amount / maxAmountRequired, network)."""

    network: str
    payee: str
    amount_raw: int  # smallest unit (raw)
    description: str = ""


def usd_to_xno_raw(price_usd: Decimal, price_xno_usd: Decimal) -> int:
    """Convert a USD invoice to the exact Nano raw amount to send.

    Rounds UP (CEILING) so the seller never under-receives, and works in Decimal
    to avoid float drift on 30-decimal amounts.
    """
    if price_xno_usd <= 0:
        raise ValueError("price_xno_usd must be positive")
    xno_needed = price_usd / price_xno_usd
    raw = (xno_needed * RAW_PER_XNO).to_integral_value(rounding=ROUND_CEILING)
    return int(raw)


async def fetch_xno_usd() -> Decimal:
    """Live XNO/USD price from a public, keyless source (CoinGecko).

    Optional in this example: swap in any pricing feed. If the network is
    unavailable the demo falls back to a published recent value so the rail math
    can still be shown. In production an agent should require >=2 independent
    pricing sources and median them (the repo's philosophy throughout).
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.coingecko.com/api/v3/simple/price?ids=nano&vs_currencies=usd"
            )
            r.raise_for_status()
            return Decimal(str(r.json()["nano"]["usd"]))
    except Exception:
        return Decimal("0.55")  # recent published fallback; clearly labelled


RPC_TIMEOUT_S = 10
RPC_MAX_BYTES = 64 * 1024


def assert_safe_rpc_url(url: str) -> None:
    """https, or plain http only to a loopback host; anything else is refused."""
    u = urlparse(url)
    if not (u.scheme == "https" or (u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1", "::1"))):
        raise ValueError("Nano RPC URL must be https (plain http only for loopback)")


async def _block_info(client: httpx.AsyncClient, url: str, block_hash: str) -> dict:
    """One bounded block_info lookup: safe URL, timeout, response size cap."""
    assert_safe_rpc_url(url)
    r = await client.post(url, json={"action": "block_info", "json_block": "true", "hash": block_hash})
    r.raise_for_status()
    if len(getattr(r, "content", b"")) > RPC_MAX_BYTES:
        raise ValueError("Nano RPC response too large")
    return r.json()


async def verify_nano_payment(
    block_hash: str,
    expected_payee: str,
    expected_amount_raw: int,
    rpc_urls: list[str] | None = None,
) -> dict:
    """UNBOUND read of a block: checks payee and amount only, and does not know which
    order the block pays for or whether it was already used. Use
    ``verify_order_payment`` to gate a service.

    Verify a Nano send on >=2 independent public RPCs. FAILS CLOSED.

    Returns a dict with ``ok``, the list of endpoints that confirmed, and the
    first reason if any endpoint refused. A payment is only considered settled
    when **all** queried nodes agree it is a confirmed ``send`` to the expected
    payee for the expected amount. One agreeing node is not enough.
    """
    rpc_urls = rpc_urls or DEFAULT_RPC_URLS
    if len(rpc_urls) < 2:
        return {"ok": False, "reason": f"need >=2 independent RPCs, got {len(rpc_urls)}"}

    confirmed: list[str] = []
    reasons: list[str] = []

    async with httpx.AsyncClient(timeout=RPC_TIMEOUT_S) as client:
        for url in rpc_urls:
            try:
                data = await _block_info(client, url, block_hash)
            except Exception as exc:  # network error, 4xx/5xx, bad JSON
                reasons.append(f"{url}: {exc}")
                continue

            # Real nodes expose the send shape as contents.type / contents.destination
            # with confirmed as the string "true". Normalize defensively.
            contents = data.get("contents", data)
            kind = contents.get("type", data.get("subtype", ""))
            destination = contents.get("destination", data.get("link_as_account", ""))
            confirmed_flag = (
                data.get("confirmed") is True or str(data.get("confirmed", "")).lower() == "true"
            )
            amount = data.get("amount", "")

            ok_node = (
                kind == "send"
                and destination == expected_payee
                and str(amount) == str(expected_amount_raw)
                and confirmed_flag
            )
            if ok_node:
                confirmed.append(url)
            else:
                reasons.append(
                    f"{url}: send={kind == 'send'}, dest_ok={destination == expected_payee}, "
                    f"amount_ok={str(amount) == str(expected_amount_raw)}, confirmed={confirmed_flag}"
                )

    return {
        "ok": len(confirmed) == len(rpc_urls),
        "confirmed_on": confirmed,
        "reasons": reasons,
    }


# ---- Order-bound verification (xno-call-binding option C + seller ledger rules) --------
# A Nano block has no memo, so "a confirmed send of the right amount exists" does not say
# which order it paid. Each order gets a random id and a commitment the buyer signs into the
# send block's ``representative``; the block hash is then claimed one-to-one with the order in
# sqlite (UNIQUE, so it survives a restart) before the service is released.
_B32 = "13456789abcdefghijkmnopqrstuwxyz"


def order_commitment(order_id: str, pay_to: str, price_raw: int) -> bytes:
    """SHA-256("xno-binding/v1/rep|" + order_id + "|" + pay_to + "|" + price_raw)."""
    msg = b"xno-binding/v1/rep|" + order_id.encode() + b"|" + pay_to.encode() + b"|" + str(int(price_raw)).encode()
    return hashlib.sha256(msg).digest()


def account_to_pubkey(addr: str) -> bytes | None:
    """32-byte public key of a nano_ account (checksum verified), or None."""
    if not isinstance(addr, str) or not addr.startswith(("nano_", "xrb_")):
        return None
    body = addr.split("_", 1)[1]
    if len(body) != 60 or any(c not in _B32 for c in body):
        return None
    n = 0
    for ch in body:
        n = n * 32 + _B32.index(ch)
    if n >> 296:
        return None
    key = (n >> 40).to_bytes(32, "big")
    if hashlib.blake2b(key, digest_size=5).digest()[::-1] != (n & ((1 << 40) - 1)).to_bytes(5, "big"):
        return None
    return key


def init_ledger(conn: sqlite3.Connection) -> None:
    conn.execute("create table if not exists nano_orders(order_id text primary key, pay_to text, "
                 "price_raw text, expires integer)")
    conn.execute("create table if not exists nano_claims(block_hash text primary key, "
                 "order_id text unique, claimed_at integer)")
    conn.commit()


def issue_order(conn: sqlite3.Connection, pay_to: str, price_raw: int, ttl_s: int = 600) -> dict:
    """Open an order with an opaque 128-bit id; give the buyer ``representative_pubkey``."""
    init_ledger(conn)
    order_id = secrets.token_hex(16)
    expires = int(time.time()) + ttl_s
    with conn:
        conn.execute("insert into nano_orders values(?,?,?,?)", (order_id, pay_to, str(price_raw), expires))
    return {"order_id": order_id, "pay_to": pay_to, "price_raw": str(price_raw), "expires": expires,
            "representative_pubkey": order_commitment(order_id, pay_to, price_raw).hex()}


def _bound_ok(data: dict, pay_to: str, price_raw: int, order_id: str) -> str:
    """Empty string if this node's answer binds the block to the order, else the reason."""
    contents = data.get("contents") or {}
    if str(data.get("confirmed")).lower() != "true":
        return "block not confirmed"
    if data.get("subtype") != "send":
        return "block is not a send"
    if contents.get("link_as_account") != pay_to:
        return "paid to wrong address"
    if str(data.get("amount")) != str(price_raw):
        return "amount does not match the order"
    rep = account_to_pubkey(contents.get("representative", ""))
    if rep is None or rep != order_commitment(order_id, pay_to, price_raw):
        return "block is not bound to this order"
    return ""


async def verify_order_payment(
    conn: sqlite3.Connection,
    order_id: str,
    block_hash: str,
    rpc_urls: list[str] | None = None,
    now: int | None = None,
) -> dict:
    """FAILS CLOSED. Releases only for a confirmed send of exactly the order's price to its
    pay_to, carrying this order's commitment, on every queried node, and whose hash is not
    claimed by another order. The claim is written before the result is returned."""
    init_ledger(conn)
    rpc_urls = rpc_urls or DEFAULT_RPC_URLS
    if len(rpc_urls) < 2:
        return {"ok": False, "reason": f"need >=2 independent RPCs, got {len(rpc_urls)}"}
    if not (isinstance(block_hash, str) and len(block_hash) == 64 and all(c in "0123456789abcdefABCDEF" for c in block_hash)):
        return {"ok": False, "reason": "invalid block hash"}
    block_hash = block_hash.upper()
    row = conn.execute("select pay_to, price_raw, expires from nano_orders where order_id=?", (str(order_id),)).fetchone()
    if row is None:
        return {"ok": False, "reason": "unknown order"}
    pay_to, price_raw, expires = row[0], int(row[1]), int(row[2])
    done = conn.execute("select block_hash from nano_claims where order_id=?", (order_id,)).fetchone()
    if done is not None:
        same = done[0] == block_hash
        return {"ok": same, "reason": "" if same else "order already paid by another block"}
    if (now if now is not None else int(time.time())) > expires:
        return {"ok": False, "reason": "order expired"}
    if conn.execute("select 1 from nano_claims where block_hash=?", (block_hash,)).fetchone():
        return {"ok": False, "reason": "block already claimed by another order"}

    async with httpx.AsyncClient(timeout=RPC_TIMEOUT_S) as client:
        for url in rpc_urls:
            try:
                data = await _block_info(client, url, block_hash)
            except Exception as exc:
                return {"ok": False, "reason": f"{url}: {exc}"}
            why = _bound_ok(data, pay_to, price_raw, order_id)
            if why:
                return {"ok": False, "reason": f"{url}: {why}"}
    try:
        with conn:  # UNIQUE(block_hash) and UNIQUE(order_id): the claim is atomic
            conn.execute("insert into nano_claims values(?,?,?)", (block_hash, order_id, int(time.time())))
    except sqlite3.IntegrityError:
        return {"ok": False, "reason": "block already claimed by another order"}
    return {"ok": True, "reason": ""}


async def main() -> None:
    print("=" * 64)
    print("Nano (XNO) settlement leg for OmniAgentPay")
    print("=" * 64)

    # 1) Map a real USD invoice to the exact raw amount.
    price_xno_usd = await fetch_xno_usd()
    invoice_usd = Decimal("0.002")  # a sub-cent per-call invoice
    amount_raw = usd_to_xno_raw(invoice_usd, price_xno_usd)
    print(f"\n1) Invoice {invoice_usd} USD at XNO/USD ≈ {price_xno_usd}")
    print(f"   -> send {amount_raw} raw ({amount_raw / RAW_PER_XNO:.8f} XNO)")

    # 2) The nano:mainnet requirements an x402 402 would carry.
    reqs = NanoRequirements(
        network="nano:mainnet",
        payee=DEMO_PAYEE,
        amount_raw=DEMO_AMOUNT_RAW,
        description="per-call access (demo)",
    )
    print(f"\n2) Parsed nano:mainnet requirement -> payee {reqs.payee[:22]}…")
    print(f"   amount {reqs.amount_raw} raw")

    # 3) Verify a real Nano send across two independent public RPCs.
    print(f"\n3) Verifying on-chain send {DEMO_BLOCK_HASH[:24]}…")
    print("   (two independent public RPCs, fail-closed; an UNBOUND read of the block -")
    print("    gate a service with verify_order_payment, which also needs the order commitment)")
    result = await verify_nano_payment(
        block_hash=DEMO_BLOCK_HASH,
        expected_payee=reqs.payee,
        expected_amount_raw=reqs.amount_raw,
    )
    if result["ok"]:
        print(f"   ✅ confirmed on {len(result['confirmed_on'])} independent RPCs")
        print("      no API key, no card, no wallet bootstrap — the open ledger is the proof")
    else:
        print(f"   ❌ not confirmed: {result['reasons']}")

    print("\n" + "=" * 64)
    print("That is the rail: settle in zero fee, confirm in ~1s, on-chain proof")
    print("readable by any node. It slots beside the USDC/x402/CCTP adapters as")
    print("another ProtocolAdapter in OmniAgentPay's router.")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
