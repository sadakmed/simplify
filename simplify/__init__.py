import os



SIMPLIFY_CACHE = os.path.expanduser(os.path.join("~/.cache", "simplify"))
if not os.path.exists(SIMPLIFY_CACHE):
    os.mkdir(SIMPLIFY_CACHE)
