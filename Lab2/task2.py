import numpy as np
'''
Cholesky decomposition for overdetermined linear system
Variant 10
'''

'''
p = norm(r)
'''


def generate_matrix(n, s):
    '''Generate (N,S) matrix A of coefficients'''
    return np.random.rand(n, s)


def column_start_ind(n):
    '''return indexes of elements that are on diagonal of L matrix
    that is this elements are the first in the column'''
    indexes = [0]*n
    indexes[0] = 1
    for i in range(1, n):
        indexes[i] = indexes[i-1] + i
    return indexes


def cholesky_spd(A, b, f):
    '''perform Cholesky decomposition
    A -- spd matrix
    f -- RHS vector
    '''
    print(A)
    n = A.shape[0]
    L = np.empty(A.shape)
    for j in range(n):
        for i in range(n):
            if i == 0:
                if j == 0:
                    L[i][j] = np.sqrt(A[i][j])
                else:
                    L[i][j] = A[i][j] / L[0][0]
            else:
                if i == j:
                    square_sum = np.sum([L[i][k]**2 for k in range(j)])
                    L[i][j] = np.sqrt(A[i][j] - square_sum)
                else:
                    prod_sum = np.sum([L[i][k]*L[j][k] for k in range(j)])
                    L[i][j] = (A[i][j] - prod_sum) / L[j][j]
            if j > i:
                L[i][j] = 0

    return L


def cholesky(A, b, f):
    '''perform Cholesky decomposition
    A -- (NxS) matrix
    b -- weight (N) vector which define diagonal (N,N) matrix with elements of b on diagonal
    f -- RHS (N) vector
    A*x=f => A.T*B*A*x=A.T*B*f
    '''
    B = np.diag(b).astype('float64')  # create diagonal matrix from vector b
    Multiplier = np.matmul(A.T, B)  # create multiplier A.T*B
    spd = np.matmul(Multiplier, A)  # spd matrix
    print("Symmetric Positive-Defined Matrix is\n", spd)
    f_hat = np.matmul(Multiplier, f)
    print("A.T*f is \n", f_hat)
    n = spd.shape[0]
    L = np.tril(np.empty(spd.shape)).astype(
        'float64')  # create lower-triagonal matrix
    # perform Cholesky decomposition
    for j in range(n):
        for i in range(n):
            if i == 0:
                if j == 0:
                    L[i][j] = np.sqrt(spd[i][j])
                else:
                    L[i][j] = spd[i][j] / L[0][0]
            else:
                if i == j:
                    square_sum = np.sum([L[i][k]**2 for k in range(j)])
                    L[i][j] = np.sqrt(spd[i][j] - square_sum)
                else:
                    prod_sum = np.sum([L[i][k]*L[j][k] for k in range(j)])
                    L[i][j] = (spd[i][j] - prod_sum) / L[j][j]
            if j > i:
                L[i][j] = 0
    print("Lower-triangular matrix is\n", L)
    print("L.T*L =\n", np.matmul(L, L.T))
    '''
    By far we've gotten, that L*L.T*x = M*f, where M=A.T*B, L -- lower-triangular form matrix
    Let L.T*x = y, then we need to solve system of matrix equations:
    L*y = M*f
    L.T*x = y
    '''
    # let's copy L matrix
    L_copy = np.copy(L)
    Lt = np.copy(L.T)
    # solve 1st equation L*y = M*f
    for i in range(n):
        f_hat[i] /= L_copy[i][i]
        L_copy[i][i] /= L_copy[i][i]
        for j in range(i+1, n):
            f_hat[j] -= f_hat[i]*L_copy[j, i]
            L_copy[j, i] = 0
    print("Now, L is:\n", L_copy)
    # solved. Now f_hat is literally y vector
    # solve 2nd equation L.T*x = y
    for i in range(n-1, -1, -1):
        f_hat[i] /= Lt[i][i]
        Lt[i][i] /= Lt[i][i]
        for j in range(i-1, -1, -1):
            f_hat[j] -= f_hat[i]*Lt[j, i]
            Lt[j, i] = 0
    print("Now, L is:\n", Lt)
    # solved. Now f_hat is literally x vector
    print("Let's check: L*L.Tx = A.T*f:\n",
          np.matmul(np.matmul(L, L.T), f_hat))
    return f_hat


def residual(A, x, f):
    res_f = np.matmul(A, x) - f
    return (res_f, np.linalg.norm(res_f))


def main_old():
    answer = input("Hello. Would you like to generate matrix? Y/N?")
    if answer == "Y":
        n = int(input("Enter the number of rows:"))
        s = int(input("Enter the number of columns:"))
        A = generate_matrix(n, s)
        print(A)
        rhs = np.random.rand(n, 1)
        cholesky(A, np.matmul(np.random.randint(100, size=n), np.eye(n)), rhs)
    else:
        n = int(input("Enter the number of rows:"))
        s = int(input("Enter the number of columns:"))
        coefs = input(
            "You are able to input coefficients by your own:").split(' ')
        A = np.array(list(map(int, coefs))).reshape((n, s))
        x = cholesky(A, np.array([1, 1, 1]), np.array([0, 0, 8]).T)
        # print(x)
        #L = cholesky_spd(coefs, np.array([16, 8]).T)
        #print(np.matmul(L, L.T))


def main():
    print("Perform Cholesky decomposition for this system:" +
          "\nx=0\ny=0\nx+2y=4\nWith weight coefficients (1,1,1), (2,2,1), (1,1,2)")
    A = np.array([1, 0, 0, 1, 2, 1]).reshape(3, 2).astype('float64')
    f = np.array([0, 0, 8]).reshape(3, 1).astype('float64')
    weights = [np.array([1, 1, 1]).astype('float64'), np.array(
        [2, 2, 1]).astype('float64'), np.array([1, 1, 2]).astype('float64')]
    for i in range(3):
        x = cholesky(A, weights[i], f)
        print(x)
        print("Residual vector and its norm:\n", residual(A, x, f))


if __name__ == "__main__":
    main()
