import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


def get_control_points():
    d = np.array([(-12.73564, 9.03455),
                  (-26.77725, 15.89208),
                  (-42.12487, 20.57261),
                  (-15.34799, 4.57169),
                  (-31.72987, 6.85753),
                  (-49.14568, 6.85754),
                  (-38.09753, -1e-05),
                  (-67.92234, -11.10268),
                  (-89.47453, -33.30804),
                  (-21.44344, -22.31416),
                  (-32.16513, -53.33632),
                  (-32.16511, -93.06657),
                  (-2e-05, -39.83887),
                  (10.72167, -70.86103),
                  (32.16511, -93.06658),
                  (21.55219, -22.31397),
                  (51.377, -33.47106),
                  (89.47453, -33.47131),
                  (15.89191, 0.00025),
                  (30.9676, 1.95954),
                  (45.22709, 5.87789),
                  (14.36797, 3.91883),
                  (27.59321, 9.68786),
                  (39.67575, 17.30712)])
    return d


def get_u_vec(end):
    u_vec = np.linspace(0, 1, end)
    u_vec[1] = u_vec[2] = u_vec[0]
    u_vec[-3] = u_vec[-2] = u_vec[-1]
    return u_vec


def hot(u_vec, u):
    return (u_vec > u).argmax() - 1


def basis_function(u, u_vec, i, k):
    if i + k - 1 >= (len(u_vec) - 2):
        u_vec_new = np.append(u_vec, u_vec[-1])
        u_vec = u_vec_new
    if k == 0:
        if u_vec[i - 1] == u_vec[i]:
            return 0
        elif u_vec[i - 1] <= u < u_vec[i]:
            return 1
        else:
            return 0
    else:
        u1 = u - u_vec[i - 1]
        u2 = u_vec[i + k - 1] - u_vec[i - 1]
        u3 = u_vec[i + k] - u
        u4 = u_vec[i + k] - u_vec[i]
        if u2 == 0 and u4 == 0:
            return np.dot(0, basis_function(u, u_vec, i, k - 1)) \
                   + np.dot(0, basis_function(u, u_vec, i + 1, k - 1))
        elif u2 == 0 or u1 == 0 and u2 == 0:
            return np.dot(0, basis_function(u, u_vec, i, k - 1)) \
                   + np.dot(u3 / u4, basis_function(u, u_vec, i + 1, k - 1))
        elif u4 == 0 or u3 == 0 and u4 == 0:
            return np.dot(u1 / u2, basis_function(u, u_vec, i, k - 1)) \
                   + np.dot(0, basis_function(u, u_vec, i + 1, k - 1))

        return np.dot(u1 / u2, basis_function(u, u_vec, i, k - 1)) + np.dot(u3 / u4,
                                                                            basis_function(u, u_vec, i + 1, k - 1))


def new_control_points(old_control_points, interpolation_points):
    de_Boor = np.zeros(len(old_control_points) - 2)
    BasisM = np.zeros((len(old_control_points) - 2, len(old_control_points) - 2))

    for i in range(len(old_control_points) - 2):
        de_Boor[i] = (u_vec2[i] + u_vec2[i + 1] + u_vec2[i + 2]) / 3

    for m in range(len(old_control_points) - 2):
        for n in range(len(old_control_points) - 2):
            NE = basis_function(de_Boor[m], u_vec2, n, 3)
            BasisM[m, n] = NE
        BasisM[-1, -1] = 1

    x = solve(BasisM, interpolation_points[:, 0])
    y = solve(BasisM, interpolation_points[:, 1])
    knots = np.zeros((len(x), 2))
    for i in range(len(x)):
        knots[i, 0] = x[i]
        knots[i, 1] = y[i]
    return knots


class CubicSpline:
    def __init__(self, u, u_vec, d):
        self.d = d
        self.u = u
        self.u_vec = u_vec
        self.su = np.zeros((2, len(u)))
        self.N = np.zeros(len(u))

        for k in range(len(u)):
            i = hot(u_vec, u[k])
            self.su[:, k] = self.blossom(u[k], i + 1, i)

    def __call__(self, *args, **kwargs):
        pass

    def blossom(self, curr_u, rm, lm):
        if rm - lm == 3:
            alpha = (self.u_vec[rm] - curr_u) / (self.u_vec[rm] - self.u_vec[lm])
            return alpha * self.d[lm, :] + (1 - alpha) * self.d[lm + 1, :]
        elif self.u_vec[rm] - self.u_vec[lm] == 0:
            alpha = 0
            return alpha * self.blossom(curr_u, rm, lm - 1) + (1 - alpha) * self.blossom(curr_u, rm + 1, lm)
        else:
            alpha = (self.u_vec[rm] - curr_u) / (self.u_vec[rm] - self.u_vec[lm])
            return alpha * self.blossom(curr_u, rm, lm - 1) + (1 - alpha) * self.blossom(curr_u, rm + 1, lm)

    def plot(self):
        plt.plot(self.su[0, :], self.su[1, :], label='Spline')
        plt.plot(self.d[:, 0], self.d[:, 1], color='r', marker='o', ls='-.', label='Control Polygon')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_basis(self, j, k):
        for i in range(len(u)):
            self.N[i] = basis_function(u[i], u_vec, j, k)
        plt.plot(self.u, self.N)
        plt.show()

    def plot_all_bases(self):
        N = np.zeros(len(u))
        S = np.zeros(len(u))
        Nx = np.zeros(len(u))
        Ny = np.zeros(len(u))
        for n in range(len(u_vec) - 2):
            for k in range(len(u)):
                N[k] = basis_function(u[k], u_vec, n, 3)
            plt.plot(u, N)
            S = S + N
            Nx = Nx + N * self.d[n, 0]
            Ny = Ny + N * self.d[n, 1]

        plt.plot(u, S)
        plt.show()

    def plot_lower_degrees(self):
        su1 = np.zeros((2, len(u)))
        su2 = np.zeros((2, len(u)))
        for k in range(len(u)):
            i = hot(u_vec, u[k])
            su2[:, k] = self.blossom(u[k], i + 1, i-1)
            su1[:, k] = self.blossom(u[k], i + 1, i-2)

        plt.plot(self.su[0, :], self.su[1, :], label='3 degree spline')
        plt.plot(su2[0, :], su2[1, :], label='2 degree spline')
        plt.plot(su1[0, :], su1[1, :], label='1 degree spline')
        plt.plot(self.d[:, 0], self.d[:, 1], color='r', marker='o', ls='-.', label='Control Polygon')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_interpolation(self):
        plt.plot(self.su[0, :], self.su[1, :], label='New Spline')
        plt.plot(knots[:, 0], knots[:, 1], color='r', marker='o', ls='-.', label='Control Polygon')
        plt.plot(interpolation_points[:, 0], interpolation_points[:, 1], color='b', marker='o', ls='-.',
                 label='Given data points')
        plt.grid()
        plt.legend()
        plt.show()


if __name__ == '__main__':
    u = np.linspace(0.001, 0.9999, 1000)
    u_vec = get_u_vec(26)
    u_vec2 = get_u_vec(24)
    control_points = get_control_points()
    interpolation_points = get_control_points()[1:-1]

    spline = CubicSpline(u, u_vec, control_points)
    #spline.plot()
    #spline.plot_basis(5, 3)
    #spline.plot_all_bases()
    #spline.plot_lower_degrees()

    knots = new_control_points(control_points, interpolation_points)
    spline2 = CubicSpline(u, u_vec2, knots)
    #spline2.plot_interpolation()
