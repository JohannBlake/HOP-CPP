import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def main():
    # Setup figure
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.35)

    # Parametric circle
    t = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), 'c--', alpha=0.5, label='Unit Circle')

    # Initial points
    point1, = ax.plot([], [], 'ro', markersize=10, label='f1(x)')
    point2, = ax.plot([], [], 'go', markersize=10, label='f2(x)')

    # Styling
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Interactive Mapping: f1, f2 -> Points on Circle")

    # Slider configuration
    ax_slider1 = plt.axes([0.2, 0.2, 0.60, 0.03])
    slider1 = Slider(
        ax=ax_slider1,
        label='f1 (val)',
        valmin=0,
        valmax=20,
        valinit=0,
        valstep=0.1
    )

    ax_slider2 = plt.axes([0.2, 0.1, 0.60, 0.03])
    slider2 = Slider(
        ax=ax_slider2,
        label='f2 (val)',
        valmin=0,
        valmax=20,
        valinit=5,
        valstep=0.1
    )

    def f_to_point(f):
        # Interpret f as angle in radians
        angle = f
        return np.cos(angle), np.sin(angle)

    def update(val):
        f1 = slider1.val
        x1, y1 = f_to_point(f1)
        point1.set_data([x1], [y1])
        
        f2 = slider2.val
        x2, y2 = f_to_point(f2)
        point2.set_data([x2], [y2])

        fig.canvas.draw_idle()

    # Register update function
    slider1.on_changed(update)
    slider2.on_changed(update)
    
    # Initial trigger
    update(0)

    plt.show()

if __name__ == "__main__":
    main()
