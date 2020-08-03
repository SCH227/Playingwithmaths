import scipy.special as sc
from scipy import integrate
from numpy import exp

yp = 40
yc = 50
alpha = 0.2
beta = 1
t = 7
delta_y = 1.4

shape = alpha*t
gamma_shape = sc.gamma(shape)
gamma_shape_1 = sc.gamma(shape-1)
gammainc_yp = sc.gammaincc(shape-1, yp)*gamma_shape_1  # because funct calculates normalized value; divided by gamma(a)
k = (beta**shape)/gamma_shape


# Creating list of values to approximate f(y-yp) and estimate the integral using trapezoidal rule
n = 50
step = (yc-yp)/n
Y = [yp+(i*step) for i in range(n+1)]


def u(y0):  # primitive found at https://www.integral-calculator.com/
    return k*((sc.gammaincc(shape-1, yp-y0)*gamma_shape_1)-gammainc_yp)


def g(x):
    return k*((x**(shape-1))*exp(-beta*x))


def G(x):
    return sc.gammaincc(shape, beta*x)  # the division by gamma_shape is included


def fy(y):
    f = lambda y0: u(y0)*g(y-y0)/(1+G(yp-y0))
    fy_yp = integrate.quad(f, yp-delta_y, yp)
    print("Error estimated for integral in ", y, " is ", fy_yp[1])   # quick check
    return fy_yp[0]


if __name__ == "__main__":
    fy_array = list(map(fy, Y))
    ps = integrate.simps(fy_array, dx=step)
    print(ps)
