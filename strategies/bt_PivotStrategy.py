import backtrader as bt
import backtrader.indicators as btind


class PivotStrategy(bt.Strategy):
    """This strategy will buy when the price is above the pivot point and sell when the price is below the pivot point.

    Args:
        bt (_type_): _description_
    """

    def __init__(self):
        pass
