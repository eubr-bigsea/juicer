# coding=utf-8
def splitter(v, delims, parse=None):
    """
    Splits a string multiple times, in a nested fashion
    """
    if delims:
        r = v.split(delims[0])
        result = []
        for x in r:
            n = splitter(x, delims[1:], parse)
            if n:
                result.append(n)
        return result
    else:
        if parse == int and v:
            return int(v)
        else:
            return v
