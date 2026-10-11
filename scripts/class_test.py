# %%
##%matplotlib widget
import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import icosphere_grid as ig


# %% class definitions


class domain(object):
    def __init__(self, x, y, z, extcoeff, albedo, g):
        self.x = x
        self.y = y
        self.z = z
        self.extcoeff = extcoeff
        self.albedo = albedo
        self.interp_extcoeff = spi.RegularGridInterpolator(
            (x, y, z), extcoeff, method="linear"
        )
        self.interp_albedo = spi.RegularGridInterpolator(
            (x, y, z), albedo, method="linear"
        )
        self.interp_g = spi.RegularGridInterpolator((x, y, z), g, method="linear")

    def __str__(self):
        return (
            f"RT-domain with {len(self.x)} x {len(self.y)} x {len(self.z)} gridpoints"
        )

    def create_grid(self):
        X, Y, Z = np.meshgrid(self.x, self.y, self.z)
        return X, Y, Z

    def get_extcoeff(self, x, y, z):
        return self.interp_extcoeff((x, y, z))

    def get_albedo(self, x, y, z):
        return self.interp_albedo((x, y, z))

    def get_g(self, x, y, z):
        return self.interp_g((x, y, z))

    # create henyey greenstein phase function
    def hg(self, theta, g):
        return (
            1
            / (4 * np.pi)
            * (1 - g**2)
            / (1 + g**2 - 2 * g * np.cos(theta)) ** (3 / 2)
        )

    # create henyey greenstein phase function
    def hg_xyz(self, theta, x, y, z):
        g = self.get_g(x, y, z)
        return (
            1
            / (4 * np.pi)
            * (1 - g**2)
            / (1 + g**2 - 2 * g * np.cos(theta)) ** (3 / 2)
        )


class periodic_domain(domain):
    def __init__(self, x, y, z, extcoeff, albedo, g):
        super().__init__(x, y, z, extcoeff, albedo, g)

    def get_extcoeff(self, x, y, z):
        if x > self.x[-1]:
            x = x - (self.x[-1] - self.x[0])
        elif x < self.x[0]:
            x = x + (self.x[-1] - self.x[0])
        if y > self.y[-1]:
            y = y - (self.y[-1] - self.y[0])
        elif y < self.y[0]:
            y = y + (self.y[-1] - self.y[0])

        print(x, y, z)
        return self.interp_extcoeff((x, y, z))

    def get_albedo(self, x, y, z):
        if x > self.x[-1]:
            x = x - (self.x[-1] - self.x[0])
        elif x < self.x[0]:
            x = x + (self.x[-1] - self.x[0])
        if y > self.y[-1]:
            y = y - (self.y[-1] - self.y[0])
        elif y < self.y[0]:
            y = y + (self.y[-1] - self.y[0])

    def get_g(self, x, y, z):
        if x > self.x[-1]:
            x = x - (self.x[-1] - self.x[0])
        elif x < self.x[0]:
            x = x + (self.x[-1] - self.x[0])
        if y > self.y[-1]:
            y = y - (self.y[-1] - self.y[0])
        elif y < self.y[0]:
            y = y + (self.y[-1] - self.y[0])

        print(x, y, z)
        return self.interp_g((x, y, z))


# %% aux functions


# create normalized three dimensional gaussian with standard deviations and means
def gauss(x, y, z, sigma_x=1, sigma_y=1, sigma_z=1, mu_x=0, mu_y=0, mu_z=0):
    '''
    Create normalized three dimensional gaussian with standard deviations and means

    Parameters
    ----------
    x : array_like
        x-coordinates of grid
    y : array_like
        y-coordinates of grid
    z : array_like
        z-coordinates of grid
    sigma_x : float, optional
        standard deviation in x-direction, by default 1
    sigma_y : float, optional
        standard deviation in y-direction, by default 1
    sigma_z : float, optional
        standard deviation in z-direction, by default 1
    mu_x : float, optional
        mean in x-direction, by default 0
    mu_y : float, optional
        mean in y-direction, by default 0
    mu_z : float, optional
        mean in z-direction, by default 0

    Returns
    -------
    array_like
        normalized three dimensional gaussian

    '''
    return (
        1
        / (2 * np.pi * sigma_x * sigma_y * sigma_z)
        * np.exp(
            -(
                (x - mu_x) ** 2 / (2 * sigma_x**2)
                + (y - mu_y) ** 2 / (2 * sigma_y**2)
                + (z - mu_z) ** 2 / (2 * sigma_z**2)
            )
        )
    )


# %% set up test domain

# create domain
x = np.linspace(-5, 5, 101)
y = np.linspace(-5, 5, 101)
z = np.linspace(0, 10, 101)

# Center of cloud
mu_x = 0.0
mu_y = 0.0
mu_z = 4.0

X, Y, Z = np.meshgrid(x, y, z)

# create extinction coefficient with integrated vertical
# maximum optical thicknes of 10
extcoeff = 2 * gauss(X, Y, Z, mu_x=mu_x, mu_y=mu_x, mu_z=mu_z)
tau_temp = np.sum(extcoeff, axis=2) * np.mean(np.diff(z))
extcoeff = extcoeff / np.max(tau_temp) * 10


albedo = gauss(X, Y, Z, mu_x=mu_x, mu_y=mu_x, mu_z=mu_z)
g = gauss(X, Y, Z, mu_x=mu_x, mu_y=mu_x, mu_z=mu_z)
g = g / (np.max(g) * 1.05)


# create domain
domain1 = periodic_domain(x, y, z, extcoeff, albedo, g)


# %% test for propagation paths 

def point_on_cuboid(theta, phi, a=1, b=2, c=3):
    """
    theta: polar angle
    phi: azimuthal angle
    a,b,c: half lengths of cuboid

    returns: point on cuboid surface
    """
    theta = theta * np.pi / 180.0
    phi = phi * np.pi / 180.0

    r = np.sqrt(a * a + b * b + c * c)
    x = r * np.sin(theta) * np.cos(phi)
    x = np.sign(x) * np.min([a, np.abs(x)])
    y = r * np.sin(theta) * np.sin(phi)
    y = np.sign(y) * np.min([b, np.abs(y)])
    z = r * np.cos(theta)
    z = np.sign(z) * np.min([c, np.abs(z)])

    return np.array([x, y, z])

#instead of using standard lat lon, we use the icosahedric grid
def make_icogrid(refinement=2):
    '''
    Create icosahedric grid with given refinement level and return vertices and triangles
    
    Parameters
    ----------
    refinement : int, optional
        refinement level, by default 2
        
    Returns
    -------
    vertices : array_like
        vertices of icosahedric grid
    triangles : array_like
        triangles of icosahedric grid

'''
    vertices, triangles = ig.make_icosahedric_grid()
    vertices, triangles = ig.refine_icosahedric_grid(vertices, triangles, refinement)
    v_sph = ig.vertices_cart2sphere(vertices)
    v_sph = v_sph * 180 / np.pi

    return v_sph

v_sph = make_icogrid(refinement=2)

P = np.zeros((3, len(v_sph)))
for i in range(np.size(v_sph, 0)):
    P[:, i] = point_on_cuboid(v_sph[i, 1], v_sph[i, 2], 2, 1, 1)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(P[0, :], P[1, :], P[2, :])
ax.set_aspect("equal", "box")
