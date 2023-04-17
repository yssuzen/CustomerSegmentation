This program uses K-Means Clustering to segment customers based on their annual income and spending score.

Requirements
  1. Python 3
  2. pandas
  3. scikit-learn
  4. matplotlib

Usage

  1. Prepare the input data in a CSV file. The file should have two columns: 'Annual Income (k$)' and 'Spending Score (1-100)'.
  2. Run the program by executing python customer_segmentation.py in the terminal.
  3. The program will show a scatter plot of the input data.
  4. The program will determine the optimal number of clusters using the elbow method and show the distortion values for different values of K.
  5. The program will segment the customers into K clusters and show a scatter plot with different colors for each cluster and the cluster centroids marked with blue X's.
