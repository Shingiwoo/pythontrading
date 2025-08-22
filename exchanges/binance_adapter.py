import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^websockets(\.|$)")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^binance(\.|$)")

_BINANCE_IMPORTED = False
Client = None
BinanceSocketManager = None
BinanceAPIException = None

def ensure():
    global _BINANCE_IMPORTED, Client, BinanceSocketManager, BinanceAPIException
    if _BINANCE_IMPORTED:
        return
    from binance.client import Client as _Client  # type: ignore
    from binance.exceptions import BinanceAPIException as _BAE
    try:
        from binance.streams import BinanceSocketManager as _BSM
    except Exception:
        try:
            from binance import ThreadedWebsocketManager as _BSM
        except Exception:
            _BSM = None
    Client = _Client
    BinanceAPIException = _BAE
    BinanceSocketManager = _BSM
    _BINANCE_IMPORTED = True
