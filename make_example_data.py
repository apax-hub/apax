import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    return x**2 + y**2


# x = np.linspace(-1.5, 1.5, num=5_000)
x = np.random.uniform(-1.5, 1.5, size=1000)
y = np.random.uniform(-1.5, 1.5, size=1000)
z = f(x, y)

print(x.shape)
print(y.shape)

# print(.shape)
print(z.shape)
# plt.scatter(x, y, c=z, s=500)
# plt.gray()

# plt.show()

plt.plot(x, y)
plt.show()
xy = np.vstack((x, y)).T
np.savez("./raw_data/parabola2D.npz", xy=xy, z=z)
# np.savez("./raw_data/parabola.npz", x=x, y=y)
