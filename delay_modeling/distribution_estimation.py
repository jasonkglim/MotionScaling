# Class for online estimation of distribution
from matplotlib import pyplot as plt
import numpy as np

def value_at_percentile(cdf, edges, percentile):
    """
    Find the value at a given percentile in the distribution.

    Parameters:
    - cdf: Cumulative distribution function (array of cumulative probabilities).
    - edges: Bin edges of the distribution.
    - percentile: Percentile value (between 0 and 100).

    Returns:
    - Value at the specified percentile.
    """
    if not (0 <= percentile <= 100):
        raise ValueError("Percentile must be between 0 and 100")

    # Find the index corresponding to the percentile in the CDF
    idx = np.searchsorted(cdf, percentile / 100.0, side='right')

    # If the percentile is beyond the range of the data, return the maximum value
    if idx == len(edges) - 1:
        return edges[-1]

    # Interpolate the value based on the surrounding bin edges
    x1, x2 = edges[idx - 1], edges[idx]
    y1, y2 = cdf[idx - 1], cdf[idx]

    # Perform linear interpolation
    interpolated_value = x1 + ((percentile / 100.0 - y1) / (y2 - y1)) * (x2 - x1)

    return interpolated_value

# calculates pmf and cdf functions from histogram pdf and edges
def calculate_pmf_cdf(pdf, edges):
    bin_width = np.diff(edges)
    cdf = np.cumsum(pdf)*bin_width
    pmf = pdf*bin_width

    return bin_width, pmf, cdf


    
class OnlineHistogram:

    def __init__(self, data=None, bin_mode='auto', window=None):
        if data is None:
            self.data = []
        else:
            self.data = data
            
        self.bin_mode = bin_mode
        if data is None:
            self.pdf = None
            self.edges = None
        else:
            self.pdf, self.edges = np.histogram(self.data, bins=bin_mode, density=True)

        if window is None:
            self.max_size = np.inf
        else:
            self.max_size = window
        #print(bin_mode, window)

    def update(self, new_data_point):
            
        if len(self.data) >= self.max_size:
            self.data.pop(0)
        #print(len(self.data))
        self.data.append(new_data_point)
        # self.bins = np.arange(min(self.data), max(self.data) + self.bin_width, self.bin_width)   
        
        self.pdf, self.edges = np.histogram(self.data, bins=self.bin_mode, density=True)
        return self.pdf, self.edges

    def plot_pmf_cdf(self, savepath):

        bin_width = np.diff(self.edges)
        cdf = np.cumsum(self.pdf)*bin_width
        pmf = self.pdf*bin_width
    
        # Calculate 90th percentile
        value90 = value_at_percentile(cdf, self.edges, 90)
    
        plt.figure(figsize=(18, 6))        
        
        # Plot the signal
        plt.subplot(1, 3, 1)
        plt.plot(self.data, marker='o')
        plt.title("Signal")
        
        # Plot the empirical CDF
        plt.subplot(1, 3, 2)
        plt.step(self.edges[:-1], cdf, where='post')
        plt.xlabel('Signal Values')
        plt.ylabel('Cumulative Probability')
        plt.title(f'Empirical CDF\n90th percentile estimated = {value90}')
        
        # Plot the PDF
        plt.subplot(1, 3, 3)
        plt.bar(self.edges[:-1], pmf, width=bin_width, alpha=0.5)
        plt.xlabel('Signal Values')
        plt.ylabel('Probability Density')
        plt.title('Empirical PDF')
        
        plt.tight_layout()
        plt.savefig(savepath)
        plt.show()
    
