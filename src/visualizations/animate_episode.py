import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle

def animate_episode(ep, env, save_path="results/visualizations/chase.mp4"):
    trace = ep.trace
    if not trace:
        print("No trace data in episode; cannot animate.")
        return

    preds = np.array([step["pred_pos"] for step in trace])
    preys = np.array([step["prey_pos"] for step in trace])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)

    # obstacles
    for ob in env.obstacles:
        c = Circle((ob.x, ob.y), ob.radius, color='gray', alpha=0.3)
        ax.add_patch(c)

    # agents
    pred_dot, = ax.plot([], [], 'ro', markersize=6, label="Predator")
    prey_dot, = ax.plot([], [], 'bo', markersize=6, label="Prey")

    # trails
    pred_line, = ax.plot([], [], 'r-', linewidth=1)
    prey_line, = ax.plot([], [], 'b-', linewidth=1)

    ax.legend()

    def init():
        pred_dot.set_data([], [])
        prey_dot.set_data([], [])
        pred_line.set_data([], [])
        prey_line.set_data([], [])
        return pred_dot, prey_dot, pred_line, prey_line

    def update(frame):
        px, py = preds[frame]
        qx, qy = preys[frame]

        
        pred_dot.set_data([px], [py])
        prey_dot.set_data([qx], [qy])

        pred_line.set_data(preds[:frame+1, 0], preds[:frame+1, 1])
        prey_line.set_data(preys[:frame+1, 0], preys[:frame+1, 1])

        return pred_dot, prey_dot, pred_line, prey_line

    ani = animation.FuncAnimation(
        fig, update, frames=len(trace),
        init_func=init, interval=30, blit=True
    )

    ani.save(save_path, fps=30, dpi=150)
    plt.close()
    print(f"Saved animation to: {save_path}")
