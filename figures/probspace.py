import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = Axes3D(fig)

# Draw x, y, and z axis markers in the same way you were in
# the code snippet in your question...
xspan, yspan, zspan = 3 * [np.linspace(0,60,20)]
zero = np.zeros_like(xspan)

ax.plot3D(xspan, zero, zero,'k--',linewidth=3.0)
ax.plot3D(zero, yspan, zero,'k--',linewidth=3.0)
ax.plot3D(zero, zero, zspan,'k--',linewidth=3.0)

ax.text(xspan.max() + 10, .5, .5, "x", color='red',fontsize=30)
ax.text(.5, yspan.max() + 10, .5, "a", color='red',fontsize=30)
ax.text(.5, .5, zspan.max() + 10, "I", color='red',fontsize=30)

# Generate and plot some random data...
ndata = 10
x = np.random.uniform(xspan.min(), xspan.max(), ndata)
y = np.random.uniform(yspan.min(), yspan.max(), ndata)
z = np.random.uniform(zspan.min(), zspan.max(), ndata)
c = np.random.random(ndata)

#ax.scatter(x, y, z, c=c, marker='o', s=20)

# This line is the only difference between the two plots above!
ax.axis("off")

plt.savefig("probspace.png")
plt.show()
