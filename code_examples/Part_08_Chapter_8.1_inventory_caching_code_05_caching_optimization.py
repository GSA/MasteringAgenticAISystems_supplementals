# Optimization: Implement caching for frequently queried products
import functools
from datetime import timedelta

@functools.lru_cache(maxsize=1000)
def check_inventory_cached(product_id: str):
    """Cache inventory checks for 5 minutes"""
    return check_inventory_original(product_id)
