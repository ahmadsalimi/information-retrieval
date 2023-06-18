from typing import Optional, List

from mir.util import getLogger

logger = getLogger(__name__)


def pickle_cache(func=None, cache_dir: Optional[str] = '/root/cache/mir', args_for_hash: Optional[List[str]] = None):
    import pickle
    import os
    import functools
    import hashlib
    import inspect
    if func is None:
        return functools.partial(pickle_cache, cache_dir=cache_dir, args_for_hash=args_for_hash)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_args = inspect.getcallargs(func, *args, **kwargs)
        args_for_hash_ = {k: v for k, v in func_args.items() if args_for_hash is None or k in args_for_hash}
        args_hash = hashlib.sha256(pickle.dumps(args_for_hash_)).hexdigest()
        cache_path = os.path.join(cache_dir, f'{func.__name__}_{args_hash}.pkl')
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(cache_path):
            logger.info(f'loading cache from {cache_path}')
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        result = func(*args, **kwargs)
        logger.info(f'saving cache to {cache_path}')
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f)
        return result
    return wrapper
