import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':
    filename = sys.argv[1]

    # Question 2
    frozen_df = pd.read_csv(filename)
    axes = frozen_df.plot(x="year", y="days")
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.yticks(np.arange(0, max(frozen_df['days']), 10))
    plt.xticks(sorted(frozen_df['year']))
    plt.xticks(rotation=90)
    plt.xticks(range(1855, 2030, 5)) 
    plt.savefig("plot.jpg")

    # Question 3a
    x = frozen_df['year'].values
    x_values = np.column_stack((np.ones_like(x), x))
    X = np.matrix(x_values)
    print("Q3a:")
    print(X)

    # Question 3b
    y = (frozen_df['days'].values)
    Y = y.astype(int)
    Y.reshape(1,-1)
    print("Q3b:")
    print(Y)

    # Question 3c
    X_T = X.T
    Z = np.dot(X_T, X)
    print("Q3c:")
    print(Z)

    # Question 3d
    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)

    # Question 3e
    PI = np.dot(I, X_T)
    print("Q3e:")
    print(PI)

    # Question 3f
    beta_hat_matrix = np.dot(PI,Y)
    hat_beta = beta_hat_matrix.tolist()[0]
    print("Q3f:")
    print(hat_beta)

    # Question 4
    x_test = 2022
    y_test = hat_beta[0] + np.dot(hat_beta[1],x_test)
    print("Q4: " + str(y_test))

    # Question 5a
    sign = None
    if hat_beta[1] > 0:
        sign = ">"
    elif hat_beta[1] < 0:
        sign = "<"
    else:
        sign = "="
    print("Q5a: " + sign)

    # Question 5b
    print("Q5b: Given that the slope is negative, it indicates an inverse correlation between the years and the duration the lake stays frozen. This means that over the years, the period the lake remains frozen has reduced. The sign signifies the regression line's slope; a < sign indicates a negative slope, suggesting that as years increase, the number of frozen days decreases.")

    # Question 6a
    x_star = (-hat_beta[0]) / hat_beta[1]
    print("Q6a: " + str(x_star))

    # Question 6b
    print("Q6b: The regression slope indicates a slight downward trend, suggesting a gradual reduction in the number of days the lake freezes. If this trend persists, there will eventually be a time when the lake doesn't freeze at all. The estimate of 400 years seems like a good estimate of when there will be no ice")

