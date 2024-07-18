import numpy as np

from .gaussian_kinematics import GaussianKinematics3D
from .find_joints import get_joints
from .IK import solveIK, solveTraj




def _get_dt(Data, dim):
    all_dt = []

    for trajectory in Data:
        traj = trajectory
        temp_velocities = np.linalg.norm(traj[dim:,:], axis=0)
        normal_indices = np.where(np.abs(temp_velocities) > 1e-8)[0]
        
        positions = traj[:dim, normal_indices]
        velocities = traj[dim:, normal_indices]
        
        dt = np.abs(np.linalg.norm(positions[:,:-1] - positions[:,1:], axis=0) / np.linalg.norm(velocities[:,:-1], axis=0))
        all_dt.append(np.mean(dt))

    return np.mean(all_dt)



def start_adapting(traj, gmm, target_start_pose, target_end_pose, dt=None, scale_ratio=None):
    """
    Require listed trajectory (no rollout) of stacked position and velocity

    # traj: an L-length list of [M, N] NumPy array: L number of trajectories, each containing M observations of N dimension

    dim: N/2 just the dimension of position

    mu: (K, dim)

    sigma: (K, dim, dim)

    pi: (K, )
    """

    pi = gmm["Prior"]
    mu = gmm["Mu"].T  #(gmm['ds_gmm'][0][0][0].T)
    sigma = gmm["Sigma"]
    dim = mu.shape[1]


    start_point = np.zeros((dim, ))
    end_point = np.zeros((dim, ))
    L = len(traj)
    for l in range(L):
        start_point += np.average(traj[l][:dim, :5], axis=1) / L
        end_point +=np.average(traj[l][:dim, -5:], axis=1) / L

    anchor_arr = get_joints(mu, sigma, end_point, start_point)[::-1]  # reverse anchor from the beginning to the end
    pi = pi[::-1]    # reverse mu, sigma, and pi, go from the beginning to the end
    mu = mu[::-1]
    sigma = sigma[::-1]

    gk = GaussianKinematics3D(pi, mu, sigma, anchor_arr)

    traj_dis = np.linalg.norm(end_point - start_point)

    if dt is None:
        dt = _get_dt(traj, dim)

    new_anchor_point = solveIK(anchor_arr, target_start_pose, target_end_pose, traj_dis, scale_ratio)  # solve for new anchor points
    _, mean_arr, cov_arr = gk.update_gaussian_transforms(new_anchor_point)
    new_gmm = {
        "Mu": mean_arr.T,
        "Sigma": cov_arr,
        "Prior": pi
    }


    plot_traj, traj_dot_arr = solveTraj(np.copy(new_anchor_point), dt)  # solve for new trajectory
    pos_and_vel = np.vstack((plot_traj[1:].T, traj_dot_arr.T))
    new_traj = [pos_and_vel]



    return new_traj, new_gmm, anchor_arr, new_anchor_point