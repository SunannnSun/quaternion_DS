from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation



def init():
    for line in lines:
        line.set_data_3d([], [], [])

    return lines


def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    #i = (2 * i) % x_t.shape[1]

    r = r_list[i]

    # line_ = np.zeros((2, 3))
    # line_[1, :] = 1
    # line_rot = r.apply(line_)

    for i, line in enumerate(lines):
        line_ = np.zeros((2, 3))
        line_[1, i] = 1
        line_rot = r.apply(line_)
        line.set_data_3d([line_rot[0, 0], line_rot[1, 0]], [line_rot[0, 1], line_rot[1, 1]], [line_rot[0, 2], line_rot[1, 2]])


    # for line, start, end in zip(lines, startpoints, endpoints):
    #     start = line_rot
    #     end = line_rot
    #     line.set_data_3d([start[0], end[0]], [start[1], end[1]], [start[2], end[2]])

    fig.canvas.draw()
    ax.set_title(f'Frame: {i}')
    return lines




fig = plt.figure()
ax = fig.add_subplot(projection="3d", proj_type="ortho")


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB

lines = [ax.plot([], [], [], c=c)[0] for c in colors]

# prepare the axes limits
ax.set_xlim((-3, 3))
ax.set_ylim((-3, 3))
ax.set_zlim((-3, 3))



r0 = R.identity()
r1 = R.from_euler("ZYX", [90, -30, 0], degrees=True)  # intrinsic
r2 = R.from_euler("zyx", [90, -30, 0], degrees=True)  # extrinsic
r_list = [r0, r1, r2]

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=2, interval=30, blit=False, repeat=False)


ax.figure.set_size_inches(10, 8)

# plt.tight_layout()
plt.show()