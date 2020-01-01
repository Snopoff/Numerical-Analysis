# Sparse matrices
# Variant 10

import numpy as np


def ind_diag(n):
    '''return indexes of elements on diagonal of a n-square matrix of special form'''
    diagonal = [0]*n

    for i in range(n):
        diagonal[i] = 0 if i == 0 else 3 if i == 1 else 7 if i == 2 else diagonal[i-1]+5
    diagonal = np.array(diagonal)

    return diagonal


def solve(A, b):
    '''solve equation Ax=b, where A -- sparse matrix of special form'''
    n = b.shape[0]
    # get indexes of diagonal elements
    diagonal = ind_diag(n)

    # eliminate 1st column
    working_row = np.array(
        [A[0], A[1], b[0]]).astype('float64')  # creating
    working_row /= A[diagonal[0]]  # dividing
    A[0], A[1], b[0] = working_row  # unpacking

    for i in range(0, n-1):
        row_below = np.array(
            [A[diagonal[i]+2], A[diagonal[i]+3], b[i+1]]).astype('float64')  # creating row below
        # elements in 1st column are 0's
        row_below -= working_row * A[diagonal[i]+2]
        A[diagonal[i]+2], A[diagonal[i]+3], b[i+1] = row_below  # unpacking

    # print('----------------------------')

    # now let's work with row from bottom to top
    # without working with 1st column cuz it's already 0's

    for i in range(n-1, 0, -1):
        # print('**************************')
        if i > 3:  # the problem is with indices: when i > 2 we take leftmost point, and don't include the element between diagonal and leftmost
            working_row = np.array([A[diagonal[i]-2], A[diagonal[i]-1],
                                    A[diagonal[i]], b[i]]).astype('float64')  # creating
            working_row /= A[diagonal[i]]  # dividing
            # main thing is here: we skip A[diagonal[i+1]-1] element
            row_above = np.array([A[diagonal[i-1]-2], A[diagonal[i-1]],
                                  A[diagonal[i-1]+1], b[i-1]]).astype('float64')
            row_above -= working_row * A[diagonal[i-1]+1]  # eliminating
            '''unpacking'''
            A[diagonal[i]-2], A[diagonal[i]-1], A[diagonal[i]], b[i] = working_row

            A[diagonal[i-1]-2], A[diagonal[i-1]
                                  ], A[diagonal[i-1]+1], b[i-1] = row_above
        elif i == 3:
            working_row = np.array([A[diagonal[i]-2], A[diagonal[i]-1],
                                    A[diagonal[i]], b[i]]).astype('float64')  # creating
            working_row /= A[diagonal[i]]  # dividing
            # difference is here:
            row_above = np.array([A[diagonal[i-1]-1], A[diagonal[i-1]],
                                  A[diagonal[i-1]+1], b[i-1]]).astype('float64')
            row_above -= working_row * A[diagonal[i-1]+1]  # eliminating
            '''unpacking'''
            A[diagonal[i]-2], A[diagonal[i]-1], A[diagonal[i]], b[i] = working_row

            A[diagonal[i-1]-1], A[diagonal[i-1]
                                  ], A[diagonal[i-1]+1], b[i-1] = row_above
        elif i == 2:
            working_row = np.array(
                [A[diagonal[i]-1], A[diagonal[i]], b[i]]).astype('float64')  # creating
            working_row /= A[diagonal[i]]  # dividing
            # difference is here:
            row_above = np.array(
                [A[diagonal[i-1]], A[diagonal[i-1]+1], b[i-1]]).astype('float64')
            row_above -= working_row * A[diagonal[i-1]+1]  # eliminating
            '''unpacking'''
            A[diagonal[i]-1], A[diagonal[i]], b[i] = working_row

            A[diagonal[i-1]], A[diagonal[i-1]+1], b[i-1] = row_above
        elif i == 1:
            working_row = np.array([A[diagonal[i]], b[i]]).astype(
                'float64')  # creating
            working_row /= A[diagonal[i]]  # dividing
            # difference is here:
            row_above = np.array(
                [A[diagonal[i-1]+1], b[i-1]]).astype('float64')
            row_above -= working_row * A[diagonal[i-1]+1]  # eliminating
            '''unpacking'''
            A[diagonal[i]], b[i] = working_row

            A[diagonal[i-1]+1], b[i-1] = row_above
        # print(A)

    #print('LAST STEP')

    # finish algorithm: eliminate those, which are non-zero and non-one's:
    for i in range(1, n-1):
        # print('**************************')
        working_row = np.array([A[diagonal[i]], b[i]]).astype('float64')
        if i > 1:
            row_below = np.array(
                [A[diagonal[i+1]-1], b[i+1]]).astype('float64')
            row_below -= working_row * A[diagonal[i+1]-1]  # eliminating
            A[diagonal[i+1]-1], b[i+1] = row_below  # unpacking
        if i == 1:
            for j in range(2, n):
                if j == 2:
                    row_below = np.array(
                        [A[diagonal[j]-1], b[j]]).astype('float64')
                    row_below -= working_row * A[diagonal[j]-1]  # eliminating
                    A[diagonal[j]-1], b[j] = row_below  # unpacking
                else:
                    row_below = np.array(
                        [A[diagonal[j]-2], b[j]]).astype('float64')
                    row_below -= working_row * A[diagonal[j]-2]  # eliminating
                    A[diagonal[j]-2], b[j] = row_below  # unpacking
        # print(A)

    # print('FINISHED!!!')
    print('Final result is: X = transpose({})'.format(b))

    return b


def ind_first_column(n):
    '''return indices of elements in 1st column'''
    inds = [0]*n
    for i in range(n):
        inds[i] = 0 if i == 0 else 2 if i == 1 else 5 if i == 2 else 9 if i == 3 else inds[i-1]+5
    inds = np.array(inds)

    return inds


def prod(A, x):
    '''return product of A and x, where A is a sparse matrix of special form'''
    n = x.shape[0]
    b = [0]*n
    first_column = ind_first_column(n)
    diagonal = ind_diag(n)
    for i in range(n):
        if i == n-1:
            # b[i] = Σ_j∈{0,1} (A[i][j]*x[j]) + Σ_j∈{n-1,n-2} (A[i][j]*x[j])
            for j in range(2):
                b[i] += A[first_column[i]+j] * x[j]
            if n == 3:  # just add last element
                b[i] += A[-1]*x[-1]
            if n > 3:  # if n > 3, then next loop won't intersect with prev. one
                for j in range(2):
                    b[i] += A[-1-j] * x[-1-j]
        elif i < 3:
            # b[i] = Σ_j=0^j=i+2 (A[i][j]*x[j])
            for j in range(i+2):
                b[i] += A[first_column[i]+j]*x[j]
        else:
            diag_ind = diagonal[i]
            for j in range(2):
                # first 2 elements of A in a row
                b[i] += A[first_column[i]+j] * x[j]
                # last 2 elements of A in a row
                b[i] += A[diag_ind+j]*x[i+j]

            b[i] += A[diag_ind-1]*x[i-1]
    return np.array(b)


def solve_pair(A, b_first, b_second):
    '''solve equations Ax=b_1, Ax=b_2, where A -- sparse matrix of special form
        return a pair (x_1, x_2)'''
    n = b_first.shape[0]
    # get indexes of diagonal elements
    diagonal = ind_diag(n)

    # eliminate 1st column
    working_row = np.array(
        [A[0], A[1], b_first[0], b_second[0]]).astype('float64')  # creating
    working_row /= A[diagonal[0]]  # dividing
    A[0], A[1], b_first[0], b_second[0] = working_row  # unpacking

    for i in range(0, n-1):
        row_below = np.array(
            [A[diagonal[i]+2], A[diagonal[i]+3], b_first[i+1], b_second[i+1]]).astype('float64')  # creating row below
        # elements in 1st column are 0's
        row_below -= working_row * A[diagonal[i]+2]
        A[diagonal[i]+2], A[diagonal[i]+3], b_first[i +
                                                    1], b_second[i+1] = row_below  # unpacking

    # print('----------------------------')

    # now let's work with row from bottom to top
    # without working with 1st column cuz it's already 0's

    for i in range(n-1, 0, -1):
        # print('**************************')
        if i > 3:  # the problem is with indices: when i > 2 we take leftmost point, and don't include the element between diagonal and leftmost
            working_row = np.array([A[diagonal[i]-2], A[diagonal[i]-1],
                                    A[diagonal[i]], b_first[i], b_second[i]]).astype('float64')  # creating
            working_row /= A[diagonal[i]]  # dividing
            # main thing is here: we skip A[diagonal[i+1]-1] element
            row_above = np.array([A[diagonal[i-1]-2], A[diagonal[i-1]],
                                  A[diagonal[i-1]+1], b_first[i-1], b_second[i-1]]).astype('float64')
            row_above -= working_row * A[diagonal[i-1]+1]  # eliminating
            '''unpacking'''
            A[diagonal[i]-2], A[diagonal[i]-1], A[diagonal[i]
                                                  ], b_first[i], b_second[i] = working_row

            A[diagonal[i-1]-2], A[diagonal[i-1]
                                  ], A[diagonal[i-1]+1], b_first[i-1], b_second[i-1] = row_above
        elif i == 3:
            working_row = np.array([A[diagonal[i]-2], A[diagonal[i]-1],
                                    A[diagonal[i]], b_first[i], b_second[i]]).astype('float64')  # creating
            working_row /= A[diagonal[i]]  # dividing
            # difference is here:
            row_above = np.array([A[diagonal[i-1]-1], A[diagonal[i-1]],
                                  A[diagonal[i-1]+1], b_first[i-1], b_second[i-1]]).astype('float64')
            row_above -= working_row * A[diagonal[i-1]+1]  # eliminating
            '''unpacking'''
            A[diagonal[i]-2], A[diagonal[i]-1], A[diagonal[i]
                                                  ], b_first[i], b_second[i] = working_row

            A[diagonal[i-1]-1], A[diagonal[i-1]
                                  ], A[diagonal[i-1]+1], b_first[i-1], b_second[i-1] = row_above
        elif i == 2:
            working_row = np.array(
                [A[diagonal[i]-1], A[diagonal[i]], b_first[i], b_second[i]]).astype('float64')  # creating
            working_row /= A[diagonal[i]]  # dividing
            # difference is here:
            row_above = np.array(
                [A[diagonal[i-1]], A[diagonal[i-1]+1], b_first[i-1], b_second[i-1]]).astype('float64')
            row_above -= working_row * A[diagonal[i-1]+1]  # eliminating
            '''unpacking'''
            A[diagonal[i]-1], A[diagonal[i]], b_first[i], b_second[i] = working_row

            A[diagonal[i-1]], A[diagonal[i-1] +
                                1], b_first[i-1], b_second[i-1] = row_above
        elif i == 1:
            working_row = np.array([A[diagonal[i]], b_first[i], b_second[i]]).astype(
                'float64')  # creating
            working_row /= A[diagonal[i]]  # dividing
            # difference is here:
            row_above = np.array(
                [A[diagonal[i-1]+1], b_first[i-1], b_second[i-1]]).astype('float64')
            row_above -= working_row * A[diagonal[i-1]+1]  # eliminating
            '''unpacking'''
            A[diagonal[i]], b_first[i], b_second[i] = working_row

            A[diagonal[i-1]+1], b_first[i-1], b_second[i-1] = row_above
        # print(A)

    #print('LAST STEP')

    # finish algorithm: eliminate those, which are non-zero and non-one's:
    for i in range(1, n-1):
        # print('**************************')
        working_row = np.array(
            [A[diagonal[i]], b_first[i], b_second[i]]).astype('float64')
        if i > 1:
            row_below = np.array(
                [A[diagonal[i+1]-1], b_first[i+1], b_second[i+1]]).astype('float64')
            row_below -= working_row * A[diagonal[i+1]-1]  # eliminating
            A[diagonal[i+1]-1], b_first[i+1], b_second[i+1] = row_below  # unpacking
        if i == 1:
            for j in range(2, n):
                if j == 2:
                    row_below = np.array(
                        [A[diagonal[j]-1], b_first[j], b_second[j]]).astype('float64')
                    row_below -= working_row * A[diagonal[j]-1]  # eliminating
                    A[diagonal[j]-1], b_first[j], b_second[j] = row_below  # unpacking
                else:
                    row_below = np.array(
                        [A[diagonal[j]-2], b_first[j], b_second[j]]).astype('float64')
                    row_below -= working_row * A[diagonal[j]-2]  # eliminating
                    A[diagonal[j]-2], b_first[j], b_second[j] = row_below  # unpacking
        # print(A)

    # print('FINISHED!!!')
    # print('Final result is: X_1 = transpose({}), X_2 = transpose({})'.format(
    #    b_first, b_second))

    return (b_first, b_second)


def number_of_elements(n):
    '''return a number of elements for certain dimension'''
    return 4*n-4 if n < 3 else 5*n-7


def error(x, x_hat, q):
    '''return error'''
    n = len(x)
    deltas = [0]*n
    for i in range(n):
        deltas[i] = abs((x[i] - x_hat[i])/x_hat[i]
                        ) if abs(x_hat[i]) > q else abs(x[i] - x_hat[i])
    return max(deltas)


def accuracy(x, x_hat):
    '''return accuracy'''
    return max((x_hat - x) / x)


def testing():
    '''tests the method in certain way'''
    dims = [10, 100, 1000]  # dimensions of a matrix
    # range of coefficients of a matrix
    coeffs = [(-10, 10), (-100, 100), (-1000, 1000)]
    np.random.seed(42)
    test_count = 1  # number of tests
    # create table
    print('№ of test | Dimension | Values | Mean error | Mean accuracy | ')
    print('-------------------------------------------------------------')
    for n in dims:
        number = number_of_elements(n)  # count the number of elements
        ones = np.ones(n)  # create identity matrix
        for coef in coeffs:
            # create random matrix in certain range
            A = np.random.uniform(*coef, number).astype('float64')
            b_ones = prod(A, ones)  # calculate rhs vector for identity matrix
            acc = 0  # mean accuracy
            err = 0  # mean error
            q = coef[1] / n
            # start testing for certain dimension and coefficients
            # perform 10 experiments
            for _ in range(10):
                x = np.random.rand(n)  # generate random matrix
                b = prod(A, x)  # calculate rhs vector for generated matrix
                x_hat = solve_pair(A, b, b_ones)[0]  # solve system
                acc += accuracy(x, x_hat)  # accumulate accuracy
                err += error(x, x_hat, q)  # acuumulate error
            acc /= 10  # get mean accuracy
            acc = np.format_float_scientific(acc, precision=3)
            err /= 10  # get mean error
            err = np.format_float_scientific(err, precision=3)
            print(test_count, n, coef, err, acc)
            print('-------------------------------------------------------------')
            test_count += 1


def main():
    # enter the dimension of a matrix
    n = int(input("Enter n -- dimension of matrix:"))
    np.random.seed(17)  # set random seed

    # count number of non-zero elements in a matrix, that is shape-like 10 variant
    # basically, 2*n -- number of elements in first 2 columns in a matrix
    # 2*(n-2) -- number of elements on the diagonal and upper diagonal excluding first 2 columns
    # n-3 -- number of elements on the lower diagonal excluding first 2 columns(for dim>2)
    # 2*n + 2*(n-2) + n-3 = 5*n-7 -- number of elements in a matrix(dim>2)

    number = number_of_elements(n)

    # create an array, that represents non-zero elements
    # elements will be set row-wise:
    # first 2 elements -- first row
    # next 3 elements -- second row
    # next 4 elements -- third row
    # next 5 elements -- fourth row
    # ---
    # last 4 elements -- nth row
    # LHS matrix of coeffs; i.e. matrix A in the eq: Ax=b
    A = np.random.rand(number).astype('float64')
    print(A)
    # RHS vector of values; i.e. vector b in the eq: Ax=b
    b = np.random.rand(n).astype('float64')
    # print(inp.shape)  # check if everything is correct
    print('---------------')
    print(b)
    print(solve(A, b))


if __name__ == "__main__":
    main()
