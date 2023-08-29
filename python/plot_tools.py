import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R



def plot_rotated_axes(ax, r , offset=(0, 0, 0), scale=1):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
    """

    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    loc = np.array([offset, offset])

    for i, (axis, c) in enumerate(zip((ax.xaxis, ax.yaxis, ax.zaxis),
                                      colors)):
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)

        line = np.zeros((2, 3))
        line[1, i] = scale
        line_rot = r.apply(line)
        line_plot = line_rot + loc
        ax.plot(line_plot[:, 0], line_plot[:, 1], line_plot[:, 2], c)

        text_loc = line[1]*1.2
        text_loc_rot = r.apply(text_loc)
        text_plot = text_loc_rot + loc[0]
        ax.text(*text_plot, axlabel.upper(), color=c,
                va="center", ha="center")
    
    plt.tight_layout()
    plt.show()


def animate_rotated_axes(ax, R_list, scale=1):
    """
    Yet to decide the input type: list of R object...for now
    """

    if 'fig' not in locals() or 'fig' not in globals():
        fig = ax.get_figure()

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axlabel = axis.axis_name
        axis.set_label_text(axlabel)


    colors = ("#FF6666", "#005533", "#1199EE")  # Colorblind-safe RGB
    lines = [ax.plot([], [], [], c=c)[0] for c in colors]


    def _init():
        for line in lines:
            line.set_data_3d([], [], [])


    def _animate(i):
        r = R_list[i-1]

        for i, (line, c) in enumerate(zip(lines, colors)):
            line_ = np.zeros((2, 3))
            line_[1, i] = scale
            line_rot_ = r.apply(line_)
            line.set_data_3d([line_rot_[0, 0], line_rot_[1, 0]], [line_rot_[0, 1], line_rot_[1, 1]], [line_rot_[0, 2], line_rot_[1, 2]])


        fig.canvas.draw()
        ax.set_title(f'Frame: {i}')


    anim = animation.FuncAnimation(fig, _animate, init_func=_init,
                                frames=len(R_list), interval=30, blit=False, repeat=True)
    
    
    plt.tight_layout()
    plt.show()






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



    animate_rotated_axes(ax, r_list)

    # plot_rotated_axes(ax, r1)


