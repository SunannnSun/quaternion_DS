import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from util import plot_tools




if __name__ == "__main__":
        
    r0 = R.identity()
    r1 = R.from_euler("ZYX", [90, -30, 0], degrees=True)  # intrinsic
    r2 = R.from_euler("zyx", [90, -30, 0], degrees=True)  # extrinsic
    r_list = [r0, r1, r2]




    fig = plt.figure()
    ax = fig.add_subplot(projection="3d", proj_type="ortho")
    ax.figure.set_size_inches(10, 8)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))
    ax.set_aspect("equal", adjustable="box")



    plot_tools.animate_rotated_axes(ax, r_list)

    # plot_rotated_axes(ax, r1)