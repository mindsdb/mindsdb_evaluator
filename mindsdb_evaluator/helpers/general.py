import inspect
from typing import Dict, Callable

def filter_fn_args(fn: Callable, args: Dict) -> Dict:
    filtered = {}
    for k, v in args.items():
        if k in inspect.signature(fn).parameters:
            filtered[k] = v
    return filtered
