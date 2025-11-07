import numpy as np

class Spiral:
    def __init__(self, n_points, n_classes, n_dimensions):
        self.N = n_points
        self.D = n_dimensions
        self.K = n_classes
        self.P = np.zeros((self.N * self.K, self.D))
        self.L = np.zeros(self.N * self.K, dtype='uint8')
        for j in range(self.K):
            ix = range(self.N * j, self.N * (j + 1))
            r = np.linspace(0.0, 1, self.N)
            t = np.linspace(j * 4, (j + 1) * 4, self.N) + np.random.randn(self.N) * 0.2
            self.P[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            self.L[ix] = j

    def generate(self):
        return self.P, self.L


class Line:
    def __init__(self, n_points, n_classes, n_dimensions):
        self.N = n_points
        self.D = n_dimensions
        self.K = n_classes
        self.P = np.zeros((self.N * self.K, self.D))
        self.L = np.zeros(self.N * self.K, dtype='uint8')
        for j in range(self.K):
            a = 2 * j - 2
            b = np.random.randn(self.N) * 2
            ix = range(self.N * j, self.N * (j + 1))
            t = np.linspace(-10, 10, self.N)
            if self.D == 2:
                self.P[ix] = np.c_[t, a * t + b]
                self.L[ix] = j

    def generate(self):
        return self.P, self.L


class Circle:
    def __init__(self, n_points, n_classes, n_dimensions):
        self.N = n_points
        self.D = n_dimensions
        self.K = n_classes
        self.P = np.zeros((self.N * self.K, self.D))
        self.L = np.zeros(self.N * self.K, dtype='uint8')
        for j in range(self.K):
            ix = range(self.N * j, self.N * (j + 1))
            r = (j + 1) * 2 + np.random.randn(self.N) * 0.5
            t = np.linspace(0, 2 * np.pi, self.N) + np.random.randn(self.N) * 0.2
            if self.D == 2:
                self.P[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
                self.L[ix] = j
            else:
                raise ValueError("Only 2D data is supported in this implementation.")

    def generate(self):
        return self.P, self.L


class Zone:
    def __init__(self, n_points, n_classes, n_dimensions):
        self.N = n_points
        self.D = n_dimensions
        self.K = n_classes
        self.P = np.zeros((self.N * self.K, self.D))
        self.L = np.zeros(self.N * self.K, dtype='uint8')
        pi = np.pi
        for j in range(self.K):
            theta = j * (2 * pi) / self.K
            a = np.cos(theta)
            b = np.sin(theta)
            ix = range(self.N * j, self.N * (j + 1))
            r = np.random.randn(self.N) * 0.5
            t = np.linspace(0, 2 * pi, self.N) + np.random.randn(self.N) * 0.2
            if self.D == 2:
                self.P[ix] = np.c_[a + r * np.sin(t), b + r * np.cos(t)]
                self.L[ix] = j
            else:
                raise ValueError("Only 2D data is supported in this implementation.")

    def generate(self):
        return self.P, self.L


class Zone_3D:
    def __init__(self, n_points, n_classes, n_dimensions, centers):
        self.N = n_points
        self.D = n_dimensions
        self.K = n_classes
        self.centers = centers
        self.P = np.zeros((self.N * self.K, self.D))
        self.L = np.zeros(self.N * self.K, dtype='uint8')
        for j in range(self.K):
            center = np.array(self.centers[j])
            ix = range(self.N * j, self.N * (j + 1))
            R = np.random.randn(self.N) * 2
            gamma = np.random.uniform(0, np.pi, self.N)
            theta = np.random.uniform(0, 2 * np.pi, self.N)
            x = R * np.sin(gamma) * np.cos(theta)
            y = R * np.sin(gamma) * np.sin(theta)
            z = R * np.cos(gamma)
            self.P[ix] = np.c_[center[0] + x, center[1] + y, center[2] + z]
            self.L[ix] = j

    def generate(self):
        return self.P, self.L


class GeneratePolynomialData:
    def __init__(self, n_points=200, coefficients=[1, -2, 3], noise_std=5):
        self.n_points = n_points
        self.coefficients = coefficients
        self.noise_std = noise_std

    def generate(self):
        x = np.linspace(-10, 10, self.n_points)
        y = sum(c * x**i for i, c in enumerate(self.coefficients[::-1]))
        y += np.random.normal(0, self.noise_std, self.n_points)
        return {"x": x, "y": y}

    def save_to_csv(self, file_name="polynomial_data.csv"):
        data = self.generate()
        combined_data = np.column_stack((data["x"], data["y"]))
        np.savetxt(file_name, combined_data, delimiter=",", header="x,y", comments="")

    def get_init_gen(self):
        equation = " + ".join(f"{c}*x^{i}" for i, c in enumerate(self.coefficients[::-1]))
        print(f"Actual polynomial: {equation}")
