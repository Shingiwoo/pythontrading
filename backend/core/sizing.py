from decimal import Decimal, ROUND_DOWN


def calculate_qty(entry: float, sl: float, equity: float, risk_per_trade: float, step_size: float) -> float:
    sl_distance = abs(entry - sl)
    risk_amount = equity * risk_per_trade
    qty = risk_amount / sl_distance
    step = Decimal(str(step_size))
    qty = Decimal(str(qty)).quantize(step, rounding=ROUND_DOWN)
    return float(qty)
