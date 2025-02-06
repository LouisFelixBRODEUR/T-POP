import matplotlib.ticker as ticker

class NoSpaceEngFormatter(ticker.EngFormatter):
    def __call__(self, x, pos=None):
        s = super().__call__(x, pos)  # Get the default formatted string
        return s.replace(' ', '')  # Remove the space