import numpy as np


def generate_spd(eigvals: np.array, hholder: np.array) -> np.array:
    '''Generate matrix with concrete eigenvalues and eigenvectors
        Parameters:
            eigvals -- eigenvalues of a matrix
            hholder -- Householder matrix wich columns are eigenvectors
    '''
    return np.matmul(np.matmul(hholder, np.diag(eigvals)), hholder.T)


def generate_eigvals(n: np.int64) -> np.array:
    '''Generate n-tuple of random numbers which are going to be eigenvalues of a matrix
        Parameters:
            n -- dimension of a matrix
    '''
    return np.random.rand(n).astype('double')


def Househoulder(n: np.int64) -> np.array:
    '''Generate (n,n) Householder matrix which is needed in order to construct a matrix with known eigvals and hholder
        Parameters:
            n -- dimension of a matrix
    '''
    w = np.random.rand(n).reshape(1, n)
    w = w / np.linalg.norm(w)
    return (np.eye(n) - 2*w*w.T).astype('double')


def power_iteration(A: np.array, num_iterations: np.int64) -> np.double:
    '''Return the greatest(in absolute value) eigenvalue of given (diagonalizable) matrix
        Parameters:
            A -- diagonalizable matrix
    '''
    x = np.random.rand(A.shape[0])  # We start from random vector
    l = np.double(0)  # That is our eigenvalue
    '''
    Iteratively, calculate:
        v = x/euclid_norm(x)
        x = A.v
        l=v.T . x
    And we gain, that
        l -> eigenvalue
        v -> +- eigenvector of l
    '''
    for _ in range(num_iterations):
        v = x/np.linalg.norm(x)
        x = np.matmul(A, v)
        l = np.matmul(v.T, x)
    return l


def inverse_iteration(A: np.array, num_iterations: np.int64) -> np.double:
    '''Return the greatest(in absolute value) eigenvalue of an inverse matrix of given (diagonalizable) matrix
        Parameters:
            A -- diagonalizable matrix
    '''
    x = np.random.rand(A.shape[0])  # We start from random vector
    l = np.double(0)  # That is am eigenvalue of an inverse matrix
    '''
    Iteratively, calculate:
        v = x/euclid_norm(x)
        x = A^-1.v  <=> A.x = v
        l=v.T . x
    And we gain, that
        l -> eigenvalue of an inverse matrix
        v -> +- eigenvector of l
    '''
    for _ in range(num_iterations):
        v = x/np.linalg.norm(x)
        x = np.linalg.solve(A, v)
        l = np.matmul(v.T, x)
    return l


def main():
    n = int(input("Hello. Please, enter the dimension of a matrix:\n"))
    K = int(
        input("Please, enter the number of iterations for methods:\n"))
    eigvals = generate_eigvals(n)
    print('generated max and min eigvals:\n', max(eigvals), min(eigvals))
    print('generated cond_number:\n', max(eigvals)/min(eigvals))
    hholder = Househoulder(n)
    print("Eigenvalues: ", eigvals)
    print("Eigenvectors are the columns of the next matrix:\n", hholder)
    A = generate_spd(eigvals, hholder)
    print("Generated matrix:\n", A)
    E = np.linalg.eig(A)
    print("Numpy computed eigvals:\n", E[0])
    print("Numpy computed max eigval and min eigval:\n",
          max(E[0]), '\n', min(E[0]))
    print("Numpy computed Cond number", max(E[0])/min(E[0]))
    eigval_max = power_iteration(A, num_iterations=K)
    eigval_min = inverse_iteration(A, num_iterations=K)
    print(eigval_max)
    print(eigval_min)
    Cond_number = eigval_max*eigval_min
    print("Condition number for given matrix is:\n", Cond_number)


def main_test():
    eigvals = np.array([1, 2, -3])
    hholder = np.array([[1, 0, -1], [1, 1, 1], [-1, 2, -1]]).T
    print("Eigenvalues: ", eigvals)
    print("Eigenvectors:\n", hholder)
    A = np.matmul(np.matmul(hholder, np.diag(eigvals)), np.linalg.inv(hholder))
    print(A)
    K = 100
    L = 100
    eigval_max = power_iteration(A, num_iterations=K)
    eigval_min = inverse_iteration(A, num_iterations=L)
    print(eigval_max)
    print(eigval_min)
    Cond_number = np.abs(eigval_max)*np.abs(eigval_min)
    print("Condition number for given matrix is:\n", Cond_number)


def test():
    print("TEST STARTED!")
    for N in [10, 100, 1000]:
        for e in [1, 10, 100]:
            cond_number_acc = []
            max_eig_acc = []
            min_eig_acc = []
            for _ in range(10):
                # f(x): (0,1) -> (a,b): x |-> a + (b - a)x
                eigvals = e-e*generate_eigvals(N)
                max_eig = max(eigvals)
                min_eig = min(eigvals)
                cond_num = max_eig/min_eig
                hholder = Househoulder(N).T
                A = generate_spd(eigvals, hholder)
                eigval_max = power_iteration(A, N*10)
                eigval_min = inverse_iteration(A, 100)
                Cond_number = eigval_max*eigval_min
                cond_number_acc.append(abs(cond_num-Cond_number))
                max_eig_acc.append(abs(max_eig-eigval_max))
                min_eig_acc.append(abs(min_eig-1/eigval_min))
            print("Numbers for N={} and e={} are:\nCond_acc = {}\nMax_acc={}\nMin_acc={}".format(
                N, e, max(cond_number_acc), max(max_eig_acc), max(min_eig_acc)))


def check():
    n = 5
    eigvals = generate_eigvals(n)
    hholder = Househoulder(n)
    A = generate_spd(eigvals, hholder)
    print(np.matmul(A, hholder[0, :]), '\n', eigvals[0]*hholder[0, :])


if __name__ == "__main__":
    main()
