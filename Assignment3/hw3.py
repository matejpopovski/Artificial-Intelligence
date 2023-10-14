from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):

    dataset = np.load(filename)

    centered_dataset = dataset - np.mean(dataset, axis=0)

    return centered_dataset

def get_covariance(dataset):

    n = len(dataset)
    transposed_dataset = np.transpose(dataset)
    dot_prod = np.dot(transposed_dataset, dataset)
    cov_mat = dot_prod / (n-1)

    return cov_mat

def get_eig(S, m):

    s_len = len(S)
    w, v = eigh(S, subset_by_index=[s_len-m, s_len-1])
    
    w = np.diag(w[::-1])

    v = np.fliplr(v)

    return w, v

def get_eig_prop(S, prop):

    w = eigh(S, eigvals_only=True)
   
    a = []
    sum_of_all_eigenvalues = np.sum(w)
    for i in range(len(S)):
        if w[i] / sum_of_all_eigenvalues > prop:
            a.append(w[i])

    x, y = eigh(S, subset_by_value=[a[0]-1, a[len(a)-1]+1])

    a = np.diag(a[::-1])
    b = np.fliplr(y)

    return a, b

def project_image(image, U):

    weights = np.dot(image, U)

    x_pca = np.dot(weights, np.transpose(U))

    return x_pca

def display_image(orig, proj):

    orig_plot = np.reshape(orig, [32, 32])
    proj_plot = np.reshape(proj, [32, 32])

    orig_plot = np.transpose(orig_plot)
    proj_plot = np.transpose(proj_plot)

    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)

    ax1.set_title('Original')
    ax2.set_title('Projection')

    val1 = ax1.imshow(orig_plot, aspect='equal')
    val2 = ax2.imshow(proj_plot, aspect='equal')

    fig.colorbar(val1, ax=ax1)
    fig.colorbar(val2, ax=ax2)
    plt.show()

    return fig, ax1, ax2