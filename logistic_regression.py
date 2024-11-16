import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import os

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def generate_ellipsoid_clusters(distance, n_samples=100, cluster_std=0.5):
    np.random.seed(0)
    covariance_matrix = np.array([[cluster_std, cluster_std * 0.8], 
                                   [cluster_std * 0.8, cluster_std]])
    
    # Generate the first cluster (class 0)
    X1 = np.random.multivariate_normal(mean=[1, 1], cov=covariance_matrix, size=n_samples)
    y1 = np.zeros(n_samples)

    # Generate the second cluster (class 1)
    X2 = np.random.multivariate_normal(mean=[1 + distance, 1 + distance], cov=covariance_matrix, size=n_samples)
    y2 = np.ones(n_samples)

    # Combine the clusters into one dataset
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def fit_logistic_regression(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    beta0 = model.intercept_[0]
    beta1, beta2 = model.coef_[0]
    return model, beta0, beta1, beta2

def do_experiments(start, end, step_num):
    shift_distances = np.linspace(start, end, step_num)
    beta0_list, beta1_list, beta2_list, slope_list, intercept_list, loss_list, margin_widths = [], [], [], [], [], [], []
    
    plt.figure(figsize=(20, 5 * step_num))

    for i, distance in enumerate(shift_distances, 1):
        X, y = generate_ellipsoid_clusters(distance=distance)
        model, beta0, beta1, beta2 = fit_logistic_regression(X, y)

        # Record coefficients and metrics
        beta0_list.append(beta0)
        beta1_list.append(beta1)
        beta2_list.append(beta2)
        slope = -beta1 / beta2
        intercept = -beta0 / beta2
        slope_list.append(slope)
        intercept_list.append(intercept)

        # Compute logistic loss
        probabilities = model.predict_proba(X)[:, 1]
        loss = -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))
        loss_list.append(loss)

        # Plot dataset
        plt.subplot(step_num, 1, i)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', alpha=0.7, label="Class 0")
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', alpha=0.7, label="Class 1")
        
        # Plot decision boundary
        x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, color='black', linestyle='--', label="Decision Boundary")
        
        # Confidence contours
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)

        contour_levels = [0.7, 0.8, 0.9]
        alphas = [0.1, 0.2, 0.3]
        for level, alpha in zip(contour_levels, alphas):
            plt.contourf(xx, yy, Z, levels=[level, 1.0], colors=['red'], alpha=alpha)
            plt.contourf(xx, yy, Z, levels=[0.0, 1 - level], colors=['blue'], alpha=alpha)

        plt.title(f"Shift Distance = {distance:.2f}")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{result_dir}/dataset.png")
    
    # Plot metrics vs shift distance
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    plt.plot(shift_distances, beta0_list, marker='o')
    plt.title("Beta0 vs Shift Distance")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta0")
    
    plt.subplot(2, 3, 2)
    plt.plot(shift_distances, beta1_list, marker='o')
    plt.title("Beta1 vs Shift Distance")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta1")
    
    plt.subplot(2, 3, 3)
    plt.plot(shift_distances, beta2_list, marker='o')
    plt.title("Beta2 vs Shift Distance")
    plt.xlabel("Shift Distance")
    plt.ylabel("Beta2")
    
    plt.subplot(2, 3, 4)
    plt.plot(shift_distances, slope_list, marker='o')
    plt.title("Slope vs Shift Distance")
    plt.xlabel("Shift Distance")
    plt.ylabel("Slope")

    plt.subplot(2, 3, 5)
    plt.plot(shift_distances, loss_list, marker='o')
    plt.title("Logistic Loss vs Shift Distance")
    plt.xlabel("Shift Distance")
    plt.ylabel("Logistic Loss")

    plt.tight_layout()
    plt.savefig(f"{result_dir}/parameters_vs_shift_distance.png")
    print("Already Save")

if __name__ == "__main__":
    start = 0.25
    end = 2.0
    step_num = 8
    do_experiments(start, end, step_num)
