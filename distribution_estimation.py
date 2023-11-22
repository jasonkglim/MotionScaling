# Class for online estimation of distribution

class OnlineEmpiricalCDF:

    def __init__(self, data=[]):
        self.data = data
        self.bin_width = 0.01
        plt.figure(figsize=(18, 6))

    def calculate_pdf_cdf(self):
        bins = np.arange(min(self.data), max(self.data) + self.bin_width, self.bin_width)
        counts, bin_edges = np.histogram(self.data, bins='auto', density=True)

        cdf = np.cumsum(counts)*self.bin_width

        return counts, bin_edges, cdf

    def update(self, new_data_point):

        self.data.append(new_data_point)
        counts, edges, cdf = self.calculate_pdf_cdf()

        plt.clf()

        # Plot the signal
        plt.subplot(1, 3, 1)
        plt.plot(self.data, marker='o')
        plt.title("Signal")

        # Plot the empirical CDF
        plt.subplot(1, 3, 2)
        plt.step(edges[:-1], cdf, where='post')
        plt.xlabel('Signal Values')
        plt.ylabel('Cumulative Probability')
        plt.title('Empirical CDF')

        # Plot the PDF
        plt.subplot(1, 3, 3)
        plt.bar(edges[:-1], counts * self.bin_width, width=self.bin_width, alpha=0.5)
        plt.xlabel('Signal Values')
        plt.ylabel('Probability Density')
        plt.title('Empirical PDF')

        plt.draw()
        plt.pause(0.0001)
