self.B_d0 = np.ones((1, 3 * (nx + 1)))
self.B_m1 = np.ones((1, 3 * (nx + 1)))
self.B_m2 = np.ones((1, 2 * (nx + 1)))
self.B_m3 = np.ones((1, 1 * (nx + 1)))
self.B_p1 = np.ones((1, 2 * (nx + 1)))
self.B_p1[1::2] = (self.inputs.gamma - 1)
self.B_p2 = np.ones((1, 1 * (nx + 1)))

self.X_d0 = np.ones((1, 4 * (nx + 1)))
self.X_m1 = np.ones((1, 3 * (nx + 1)))
self.X_m2 = np.ones((1, 2 * (nx + 1)))
self.X_m3 = np.ones((1, 1 * (nx + 1)))
self.X_p1 = np.ones((1, 2 * (nx + 1)))
self.X_p2 = np.ones((1, 2 * (nx + 1)))

self.Xi_d0 = np.ones((1, 3 * (nx + 1)))
self.Xi_m1 = np.ones((1, 2 * (nx + 1)))
self.Xi_m2 = np.ones((1, 2 * (nx + 1)))
self.Xi_m3 = np.ones((1, 1 * (nx + 1)))
self.Xi_p1 = np.ones((1, 3 * (nx + 1)))
self.Xi_p2 = np.ones((1, 2 * (nx + 1)))
self.Xi_p3 = np.ones((1, 1 * (nx + 1)))

self.lam = np.zeros((1, 4 * (nx + 1)))