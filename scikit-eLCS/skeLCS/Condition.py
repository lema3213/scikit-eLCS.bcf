class Condition:
    def __init__(self, cf=None, max_depth=None, value=None):
        if cf is not None and max_depth is not None:
            # First initialization approach
            expression = str(cf)
            tokens = expression.split()
            self.expression = expression
            self.max_depth = max_depth
            self.codeFragment = cf
            self.positions = sorted([int(tk[1:]) for tk in tokens if tk[0]=='D'])
            self.value = value
            self.is_dc = False
        else:
            # Second initialization approach (default)
            self.expression = 'dc'
            self.positions = []
            self.max_depth = None
            self.codeFragment = None
            self.is_dc = True

        if self.expression == "dc":
            self._str_cache = "dc"
        else:
            self._str_cache = self.expression + "|" + str(value)


    def __str__(self):
        return self._str_cache
