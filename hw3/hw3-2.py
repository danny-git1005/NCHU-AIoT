import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import LinearSVC
import streamlit as st

# Add title to the Streamlit app
st.title('3D Scatter Plot with Separating Hyperplane')


# Step 1: Generate 600 random points centered at (0, 0) with variance 10
np.random.seed(0)
num_points = 600
mean = 0
variance = 10
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

# Streamlit slider to adjust distance threshold
distance_threshold = st.slider('Distance Threshold', min_value=0.0, max_value=10.0, value=4.0, step=0.1)

# Calculate distances from the origin
distances = np.sqrt(x1**2 + x2**2)

# Assign labels Y=0 for points within distance threshold, Y=1 for the rest
Y = np.where(distances < distance_threshold, 0, 1)

# Step 2: Calculate x3 as a Gaussian function of x1 and x2
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

x3 = gaussian_function(x1, x2)

# Step 3: Train a LinearSVC to find a separating hyperplane
X = np.column_stack((x1, x2, x3))
clf = LinearSVC(random_state=0, max_iter=10000)
clf.fit(X, Y)
coef = clf.coef_[0]
intercept = clf.intercept_

# Create 3D scatter plot with Y as color and the separating hyperplane
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1[Y==0], x2[Y==0], x3[Y==0], c='blue', marker='o', label='Y=0')
ax.scatter(x1[Y==1], x2[Y==1], x3[Y==1], c='red', marker='s', label='Y=1')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('3D Scatter Plot with Y Color and Separating Hyperplane')
ax.legend()

# Create a meshgrid to plot the separating hyperplane
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                     np.linspace(min(x2), max(x2), 10))
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]
ax.plot_surface(xx, yy, zz, color='gray', alpha=0.5)

# Display the plot in Streamlit
st.pyplot(fig)