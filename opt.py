import math


def phi(x):
    return math.exp(-x) + pow(x, 2)


def initialize_probes(a, b, alpha):
    lam = a + (1 - alpha) * (b - a)
    miu = a + alpha * (b - a)
    return lam, miu, phi(lam), phi(miu)


def solve(a, b, alpha, epsilon):
    print("alpha:{}, epsilon:{}".format(alpha, epsilon))
    lam, miu, phi_lam, phi_miu = initialize_probes(a, b, alpha)
    print("initial: a={}, b={}, lambda={}, miu={}, phi_lam={}, phi_miu={}"
          .format(a, b, lam, miu, phi_lam, phi_miu))
    i = 1
    while b - a > epsilon:
        if phi_lam <= phi_miu:
            b = miu
            miu = lam
            phi_miu = phi_lam
            lam = a + b - miu
            phi_lam = phi(lam)
        else:
            a = lam
            lam = miu
            phi_lam = phi_miu
            miu = b - lam + a
            phi_miu = phi(miu)
        if lam >= miu:
            print("error is too large, need to re-compute")
            lam, miu, phi_lam, phi_miu = initialize_probes(a, b, alpha)
        print("iteration={}: a={}, b={}, lambda={}, miu={}, phi_lam={}, phi_miu={}"
            .format(i, a, b, lam, miu, phi_lam,phi_miu))
        i += 1
    print("Final result: optimized x={}, optimal value={}".format((a + b) / 2, phi((a + b) / 2)))
    return

if __name__ == '__main__':
    a, b = 0, 1
    epsilon = 1e-4
    alpha1 = 0.618
    alpha2 = (math.sqrt(5) - 1) / 2
    solve(a, b, alpha1, epsilon)
    solve(a, b, alpha2, epsilon)

