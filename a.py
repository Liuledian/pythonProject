def phi(miu):
    return 3*(miu-4)**4-4*(miu-4)**3-12*(miu-4)**2

def phi_dot(miu):
    return 12*(miu-4)**3 - 12*(miu-4)**2 - 24*(miu-4)

def bound(alpha, miu):
    return 832 + alpha * (-864) * miu