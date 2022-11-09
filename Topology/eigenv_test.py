import numpy as np
import itertools

np.set_printoptions(edgeitems=5, linewidth=120, formatter=dict(float=lambda x: "% .4f" % x))


def calc_lambda(A):
    """Calculate the current lambda_2 and K_lambda_2 and return them."""
    D = np.diag(A.sum(1))
    L = D - A
    lambdas, vectors = np.linalg.eig(L)
    sort = lambdas.argsort()
    l2 = lambdas[sort][1]

    f = vectors[:,sort][:,1]
    K_l2 = min((a - b) ** 2 for a, b in itertools.combinations(f, 2) if abs(a - b) > 10e-5)

    return l2, K_l2


if __name__ == '__main__':
    # A = np.ones(5) - np.eye(5)
    A = np.array([[0, 1, 0, 1, 0],
                  [1, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 1],
                  [0, 1, 0, 1, 0]])
    A = A - np.multiply(np.abs(np.random.normal(0, 0.2, np.shape(A))), A)
    L = np.diag(A.sum(1)) - A

    print(f"A={np.array2string(A, prefix='A=')}\n")
    print(f"L={np.array2string(L, prefix='L=')}")
    print("================\n")

    lambdas, vectors = np.linalg.eig(L)
    print(f"lambdas={np.array2string(lambdas, prefix='lambdas=')}")
    print(f"vectors={np.array2string(vectors, prefix='vectors=')}")
    print("================\n")

    print("L*f == v*f")
    for i in range(5):
        print('\t', abs(L @ vectors[:, i] - lambdas[i] * vectors[:, i]))
    print("================\n")

    print("Min square difference between elements of f")
    for k in range(5):
        mink = 100
        for i in range(5):
            for j in range(5):
                if abs(vectors[i, k] - vectors[j, k]) > 0.0001:
                    d = (vectors[i, k] - vectors[j, k]) ** 2
                    mink = min(d, mink)
        print('\t', k, mink)
    print("================\n")

    print("Sorted eigenvalues and eigenvectors.")
    sort = lambdas.argsort()
    lambdas = lambdas[sort]
    vectors = vectors[:,sort]
    print(f"sort=   {np.array2string(sort, prefix='sort=      ')}")
    print(f"lambdas={np.array2string(lambdas, prefix='lambdas=')}")
    print(f"vectors={np.array2string(vectors, prefix='vectors=')}")
    print("================\n")

    print("Final result")
    l2, kl2 = calc_lambda(A)
    print(f"\t{l2=}, {kl2=}")
