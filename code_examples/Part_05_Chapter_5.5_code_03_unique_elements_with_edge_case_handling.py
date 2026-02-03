def unique_elements(lst):
    if not lst:
        return []
    return list(dict.fromkeys(lst))
