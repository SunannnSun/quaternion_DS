from scipy.spatial.transform import Rotation as R
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.animation as animation



def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.
    https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set(xlim=(x - radius, x + radius), ylim=(y - radius, y + radius), zlim=(z - radius, z + radius))




def plot_rotated_axes(ax, r, name=None, offset=(0, 0, 0), scale=1):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    """
    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    loc = np.array([offset, offset])
    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                      colors)):
        axlabel = axis.axis_name
        # axis.set_label_text(axlabel)
        # axis.label.set_color(c)
        # axis.line.set_color(c)
        # axis.set_tick_params(colors=c)
        line = np.zeros((2, 3))
        line[1, i] = scale
        line_rot = r.apply(line)
        line_plot = line_rot + loc
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)
        # text_loc = line[1]*1.2
        # text_loc_rot = r.apply(text_loc)
        # text_plot = text_loc_rot + loc[0]
        # ax.text(*text_plot, axlabel.upper(), color=c,
        #         va="center", ha="center")
    # ax.text(*offset, name, color="k", va="center", ha="center",
    #         bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle"})
    


if __name__ == "__main__":
        
    r0 = R.identity()
    r1 = R.from_euler("ZYX", [90, -30, 0], degrees=True)  # intrinsic
    r2 = R.from_euler("zyx", [90, -30, 0], degrees=True)  # extrinsic
    r = [r0, r1, r2]

    ax = plt.figure().add_subplot(projection="3d", proj_type="ortho")

    plot_rotated_axes(ax, r0, name="r0", offset=(0, 0, 0))
    plot_rotated_axes(ax, r1, name="r1", offset=(0, 0, 0))
    # plot_rotated_axes(ax, r2, name="r2", offset=(6, 0, 0))

    set_axes_equal(ax)
    ax.set_aspect("equal", adjustable="box")
    ax.figure.set_size_inches(10, 8)
    plt.tight_layout()


    # def update(frame):    
    #     ax.clear()      
    #     plot_rotated_axes(ax, r[frame-1], name="r", offset=(0, 0, 0))
 
    #     ax.set_title(f'Frame: {frame}')


    # fig = plt.figure()
    # # ax = fig.add_subplot(projection="3d", proj_type="ortho")
    # ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    # ani = animation.FuncAnimation(fig, update, frames=2, interval=60, repeat=True)
    # ax.set_xlim((-8, 8))
    # ax.set_ylim((-8, 8))
    # ax.set_zlim((-8, 8))    
    # # set_axes_equal(ax)
    # # ax.set_aspect("equal", adjustable="box")
    # ax.figure.set_size_inches(10, 8)
    # plt.tight_layout()

    plt.show()
