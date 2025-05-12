import numpy as np

# 系統定義
A = np.array([
    [4, -1,  0, -1,  0,  0],
    [-1, 4, -1,  0, -1,  0],
    [0, -1,  4,  0,  1, -1],
    [-1, 0,  0,  4, -1, -1],
    [0, -1,  0, -1,  4, -1],
    [0, 0, -1,  0, -1,  4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

# 共用參數
x0 = np.zeros(6)
epsilon = 1e-6
max_iter = 1000

# 1. Jacobi Method
def jacobi(A, b, x0, eps, max_iter):
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.linalg.inv(D)
    x = x0.copy()
    for i in range(max_iter):
        x_new = D_inv @ (b - R @ x)
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            return x_new, i + 1
        x = x_new
    return x, max_iter

# 2. Gauss-Seidel Method
def gauss_seidel(A, b, x0, eps, max_iter):
    x = x0.copy()
    n = len(b)
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            return x_new, k + 1
        x = x_new
    return x, max_iter

# 3. SOR Method
def sor(A, b, x0, omega, eps, max_iter):
    x = x0.copy()
    n = len(b)
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            return x_new, k + 1
        x = x_new
    return x, max_iter

# 4. Conjugate Gradient Method
def conjugate_gradient(A, b, x0, eps, max_iter):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    for k in range(max_iter):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) < eps:
            return x, k + 1
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
    return x, max_iter

# 執行與印出結果
jacobi_sol, jacobi_iter = jacobi(A, b, x0, epsilon, max_iter)
gs_sol, gs_iter = gauss_seidel(A, b, x0, epsilon, max_iter)
sor_sol, sor_iter = sor(A, b, x0, omega=1.25, eps=epsilon, max_iter=max_iter)
cg_sol, cg_iter = conjugate_gradient(A, b, x0, epsilon, max_iter)

print("Jacobi solution:", jacobi_sol, "in", jacobi_iter, "iterations")
print("Gauss-Seidel solution:", gs_sol, "in", gs_iter, "iterations")
print("SOR solution (ω=1.25):", sor_sol, "in", sor_iter, "iterations")
print("Conjugate Gradient solution:", cg_sol, "in", cg_iter, "iterations")
