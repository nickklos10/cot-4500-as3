import numpy as np
from scipy.linalg import lu


# Differential equation function
def f(t, y):
  return t - y**2


# Euler Method
def euler_method(f, y0, t0, t1, n):
  h = (t1 - t0) / n
  t = t0
  y = y0
  for _ in range(n):
    y += h * f(t, y)
    t += h
  return y


# Runge-Kutta Method
def runge_kutta_method(f, y0, t0, t1, n):
  h = (t1 - t0) / n
  t = t0
  y = y0
  for _ in range(n):
    k1 = h * f(t, y)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
    k4 = h * f(t + h, y + k3)
    y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    t += h
  return y


# Gaussian Elimination
def gaussian_elimination(augmented_matrix):
  n = len(augmented_matrix)
  for i in range(n):
    if augmented_matrix[i][i] == 0:
      for j in range(i + 1, n):
        if augmented_matrix[j][i] != 0:
          augmented_matrix[i], augmented_matrix[j] = augmented_matrix[
              j], augmented_matrix[i]
          break
    for j in range(i + 1, n):
      if augmented_matrix[j][i] == 0:
        continue
      ratio = augmented_matrix[j][i] / augmented_matrix[i][i]
      for k in range(n + 1):
        augmented_matrix[j][k] -= ratio * augmented_matrix[i][k]
  return augmented_matrix


# Backward Substitution
def backward_substitution(augmented_matrix):
  n = len(augmented_matrix)
  solution = [0 for _ in range(n)]
  for i in range(n - 1, -1, -1):
    solution[i] = augmented_matrix[i][n] / augmented_matrix[i][i]
    for j in range(i - 1, -1, -1):
      augmented_matrix[j][n] -= augmented_matrix[j][i] * solution[i]
  return solution


# LU Factorization
def lu_factorization(matrix):
  P, L, U = lu(matrix)
  determinant = np.linalg.det(U)
  return L, U, determinant


# Check if matrix is diagonally dominant
def is_diagonally_dominant(matrix):
  for i in range(len(matrix)):
    sum_row = sum(abs(matrix[i][j]) for j in range(len(matrix)) if i != j)
    if abs(matrix[i][i]) <= sum_row:
      return False
  return True


# Check if matrix is positive definite
def is_positive_definite(matrix):
  try:
    np.linalg.cholesky(matrix)
    return True
  except np.linalg.LinAlgError:
    return False


# Initial conditions for Euler and Runge-Kutta
y0 = 1  # initial y value
t0 = 0  # start of interval
t1 = 2  # end of interval
n = 10  # number of iterations

# Euler and Runge-Kutta solutions
euler_solution = euler_method(f, y0, t0, t1, n)
runge_kutta_solution = runge_kutta_method(f, y0, t0, t1, n)

# Augmented matrix for Gaussian elimination
augmented_matrix = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]])

# Gaussian elimination and backward substitution
reduced_matrix = gaussian_elimination(augmented_matrix.copy())
solution = backward_substitution(reduced_matrix)

# Matrix for LU Factorization
lu_matrix = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2],
                      [-1, 2, 3, -1]])

# LU Factorization
L, U, determinant = lu_factorization(lu_matrix)

# Matrices for checking properties
diagonal_dominant_matrix = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1],
                                     [0, 1, 7, 2, 3], [4, 2, 3, 12, 2],
                                     [3, 2, 4, 0, 8]])

positive_definite_matrix = np.array([[2, 2, 1], [2, 3, 0], [1, 0, 2]])

# Check matrix properties
is_dd = is_diagonally_dominant(diagonal_dominant_matrix)
is_pd = is_positive_definite(positive_definite_matrix)

# Print results
print(f"Euler Method Solution: {euler_solution}")
print(f"Runge-Kutta Method Solution: {runge_kutta_solution}")
print(f"Gaussian Solution: {solution}")
print(f"L Matrix:\n{L}")
print(f"U Matrix:\n{U}")
print(f"Determinant: {determinant}")
print(f"Is Diagonally Dominant: {is_dd}")
print(f"Is Positive Definite: {is_pd}")
