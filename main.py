import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from src.util import load_tools, process_tools, quat_tools


from mpl_toolkits.mplot3d import art3d
from matplotlib.patches import Circle, PathPatch, Rectangle

import matplotlib.patches as mpatches


def rotation_matrix(d):
    sin_angle = np.linalg.norm(d)
    if sin_angle == 0:return np.identity(3)
    d /= sin_angle
    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M

def pathpatch_2d_to_3d(pathpatch, z, normal):
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised
    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color    

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector    
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def pathpatch_translate(pathpatch, delta):
    pathpatch._segment3d += delta

def plot_plane(ax, point, normal, size=10, color='y'):    
    # p = Circle((0, 0), size, facecolor = color, alpha = .2)

    p = mpatches.RegularPolygon((0, 0), 4, radius=size, orientation=0, color='gray', alpha=0.1)

    # p = Rectangle((0, 0), width = 1 , height = 1, facecolor = color, alpha = .2)

    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z=0, normal=normal)
    pathpatch_translate(p, (point[0], point[1], point[2]))

    # pass



def plot_unit_sphere(ax):
    # Parametric equations for a sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Plot the sphere
    ax.plot_surface(x, y, z, color=(0, 0, 0, 0),  edgecolor=(0, 0, 0, 0.05),  rstride=8, cstride=8)



def plot_points_on_sphere(ax, points, color='r'):
    # Plot 3D points on the sphere
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    ax.scatter(x, y, z, color=color, s=0.5, label='Data Points')




def plot_tangent_plane(ax, v):
   orig = np.zeros((3, ))
   v=np.array(v)
   ax.quiver(orig[0], orig[1], orig[2], v[0], v[1], v[2],color='r')




# Example usage:
# Generate random points on the unit sphere
np.random.seed(0)
num_points = 100
points = np.random.randn(num_points, 3)
points /= np.linalg.norm(points, axis=1)[:, np.newaxis]  # Normalize to unit length

# Create a 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the unit sphere
plot_unit_sphere(ax)

# Plot the points on the sphere

# x, x_dot, x_att, x_init = load_tools.load_data(2)
p_raw, q_raw, t_raw = load_tools.load_clfd_dataset(task_id=2, num_traj=5, sub_sample=1)

p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)
p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)


p_in /= np.linalg.norm(p_in, axis=1, keepdims=True)

plot_points_on_sphere(ax, p_in)


p_in_att = quat_tools.riem_log(p_in[-1, :], p_in)




plot_points_on_sphere(ax, p_in_att+p_in[-1, :], color='k')




# plot_tangent_plane(ax, p_in[-1, :])


plot_plane(ax,  p_in[-1, :], p_in[-1, :], size=2)    


# Set plot labels and legend
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Points on Unit Sphere') 

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
# ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
# ax.zaxis.set_major_locator(MaxNLocator(nbins=4))


# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

ax.set_axis_off()
# ax.legend()

ax.set_aspect('equal')
# Show plot
plt.show()
