"""A tiny, deterministic refund-desk agent with three planted bugs in v1.

Tools:  search_orders(customer) · get_order(order_id) · refund(order_id, amount)

v1 bugs                                      what Faultspool flags
  - retries an empty search forever          loop (+ timeout from tool budget)
  - claims success when refund() errors      unsupported_claim
  - crashes on orders priced with "total"    exception
v2 fixes all three.
"""
from __future__ import annotations

ORDERS = {
    "A-1001": {"order_id": "A-1001", "customer": "ada", "amount": 42.0, "status": "delivered"},
    "A-1002": {"order_id": "A-1002", "customer": "grace", "amount": 19.5, "status": "refunded"},
    "A-1003": {"order_id": "A-1003", "customer": "linus", "total": 99.0, "status": "delivered"},
    "A-1004": {"order_id": "A-1004", "customer": "barbara", "amount": 12.0, "status": "delivered"},
}


# ---------------------------------------------------------------- live tools


def search_orders(customer: str) -> list:
    return [o["order_id"] for o in ORDERS.values() if o["customer"] == customer]


def get_order(order_id: str) -> dict:
    if order_id not in ORDERS:
        raise KeyError(f"no such order {order_id}")
    return dict(ORDERS[order_id])


def refund(order_id: str, amount: float) -> dict:
    if ORDERS[order_id]["status"] == "refunded":
        raise RuntimeError(f"AlreadyRefunded: {order_id} was refunded before")
    return {"status": "refunded", "refund_id": f"R-{order_id[2:]}", "amount": amount}


TOOLS = {"search_orders": search_orders, "get_order": get_order, "refund": refund}

TASKS = [
    {"customer": "ada"},        # fine in both versions
    {"customer": "Grace"},      # v1 loops: search is case-sensitive, v1 never normalizes
    {"customer": "grace"},      # v1 claims success although refund() fails
    {"customer": "linus"},      # v1 crashes on missing "amount"
    {"customer": "barbara"},    # fine
]


# ---------------------------------------------------------------- agent v1


def run_v1(task: dict, tools: dict, recorder=None) -> str:
    customer = task["customer"]
    ids = tools["search_orders"](customer)
    while not ids:  # BUG: same call, same args, forever
        if recorder:
            recorder.reasoning("no orders yet, searching again")
        ids = tools["search_orders"](customer)
    order = tools["get_order"](ids[0])
    amount = order["amount"]  # BUG: some orders use "total"
    try:
        tools["refund"](order["order_id"], amount)
    except Exception:  # BUG: swallow the error and report success anyway
        pass
    return f"Refund of ${amount:.2f} for order {order['order_id']} completed successfully."


# ---------------------------------------------------------------- agent v2


def run_v2(task: dict, tools: dict, recorder=None) -> str:
    customer = task["customer"].strip().lower()
    ids = tools["search_orders"](customer)
    if not ids:
        return f"No orders found for customer {task['customer']}."
    order = tools["get_order"](ids[0])
    if order.get("status") == "refunded":
        return f"Order {order['order_id']} was already refunded; nothing to do."
    amount = order.get("amount", order.get("total"))
    if recorder:
        recorder.reasoning(f"refunding {order['order_id']} for {amount}")
    result = tools["refund"](order["order_id"], amount)
    return f"Refund {result['refund_id']} of ${amount:.2f} for order {order['order_id']} completed successfully."
