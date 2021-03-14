import zipline
from zipline.api import order, record, symbol

def initialize(contest):
    pass


def handle_data(context, data):
    order(symbol('AAPL'), 5)
    record(AAPL=data.current(symbol('AAPL', "price")))
