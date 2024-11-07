from matplotlib import pyplot as plt
import numpy as np


def value_at_percentile(cdf, edges, percentile):
    """
    Find the value at a given percentile in the distribution.

    Parameters:
    - cdf: Cumulative distribution function (array of cumulative probabilities)
    - edges: Bin edges of the distribution.
    - percentile: Percentile value (between 0 and 100).

    Returns:
    - Value at the specified percentile.
    """
    if not (0 <= percentile <= 100):
        raise ValueError("Percentile must be between 0 and 100")

    idx = np.searchsorted(cdf, percentile / 100.0, side="right")
    if idx == len(edges) - 1:
        return edges[-1]

    # Linear interpolation
    x1, x2 = edges[idx - 1], edges[idx]
    y1, y2 = cdf[idx - 1], cdf[idx]
    return x1 + ((percentile / 100.0 - y1) / (y2 - y1)) * (x2 - x1)


def calculate_pmf_cdf(pdf, edges):
    """
    Calculate the PMF and CDF from a histogram PDF and bin edges.

    Parameters:
    - pdf: Probability density function values.
    - edges: Bin edges of the histogram.

    Returns:
    - bin_width: Width of each bin.
    - pmf: Probability mass function values.
    - cdf: Cumulative distribution function values.
    """
    bin_width = np.diff(edges)
    cdf = np.cumsum(pdf * bin_width)
    pmf = pdf * bin_width
    return bin_width, pmf, cdf


class DistributionEstimation:
    """
    Online histogram estimation class using a sliding window to estimate
    the distribution as an empirical histogram.
    """

    def __init__(self, data=None, bin_mode="auto", window=100):
        """
        Initialize the distribution estimation object.

        Parameters:
        - data (list): Initial data points.
        - bin_mode (str or int): Mode for binning the histogram.
            If int, represents the number of bins.
        - window (int): Maximum number of data points to store.
        """
        self.data = [] if data is None else data
        self.bin_mode = bin_mode
        self.max_size = window
        self.pdf, self.edges = (
            self._calculate_pdf_edges() if data else (None, None)
        )

    def _calculate_pdf_edges(self):
        """Helper to calculate PDF and edges for the histogram."""
        return np.histogram(self.data, bins=self.bin_mode, density=True)

    def update(self, new_data_point):
        """Add a new data point and update the histogram."""
        if len(self.data) >= self.max_size:
            self.data.pop(0)
        self.data.append(new_data_point)
        self.pdf, self.edges = self._calculate_pdf_edges()
        return self.pdf, self.edges

    def get_value_at_percentile(self, percentile):
        """
        Return the value at a specified percentile in the current estimated
        distribution.

        Parameters:
        - percentile (int): Percentile value (0-100).

        Returns:
        - Value at the specified percentile.
        """
        cdf = np.cumsum(self.pdf * np.diff(self.edges))
        return value_at_percentile(cdf, self.edges, percentile)

    def plot_pmf_cdf(self, savepath):
        """
        Plot the signal, empirical CDF, and empirical PDF,
        saving the figure to a file.

        Parameters:
        - savepath (str): Path to save the plot.
        """
        bin_width, pmf, cdf = calculate_pmf_cdf(self.pdf, self.edges)
        value90 = value_at_percentile(cdf, self.edges, 90)

        plt.figure(figsize=(18, 6))

        # Signal plot
        plt.subplot(1, 3, 1)
        plt.plot(self.data, marker="o")
        plt.title("Signal")

        # Empirical CDF plot
        plt.subplot(1, 3, 2)
        plt.step(self.edges[:-1], cdf, where="post")
        plt.xlabel("Signal Values")
        plt.ylabel("Cumulative Probability")
        plt.title(f"Empirical CDF\n90th percentile = {value90:.2f}")

        # PDF plot
        plt.subplot(1, 3, 3)
        plt.bar(self.edges[:-1], pmf, width=bin_width, alpha=0.5)
        plt.xlabel("Signal Values")
        plt.ylabel("Probability Density")
        plt.title("Empirical PDF")

        plt.tight_layout()
        plt.savefig(savepath)
        plt.show()
