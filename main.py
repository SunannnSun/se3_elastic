import numpy as np
import matplotlib.pyplot as plt

from src.util import plot_tools, load_tools, process_tools, quat_tools



# Create a 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plot_tools.plot_unit_sphere(ax)


# load and normalize data
p_raw, q_raw, t_raw          = load_tools.load_clfd_dataset(task_id=2, num_traj=5, sub_sample=1)

p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)
p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)

p_in /= np.linalg.norm(p_in, axis=1, keepdims=True)


# plot 
plot_tools.plot_points_on_sphere(ax, p_in)

p_in_att = quat_tools.riem_log(p_in[-1, :], p_in)

plot_tools.plot_points_on_sphere(ax, p_in_att+p_in[-1, :], color='k')

plot_tools.plot_plane(ax,  p_in[-1, :], p_in[-1, :], size=2)    

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.set_axis_off()

ax.set_aspect('equal')

plt.show()
