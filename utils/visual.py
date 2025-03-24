import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R
import os


def vis_scene(
    eye2base_poses: np.ndarray,
    hand2base_poses: np.ndarray,
    point_cloud=None,
    point_colors=None,
    output_path="./hec_results.html",
):
    # Create a subplot with a 3D scene
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scene"}]],
        subplot_titles=("3D Trajectory & Point Cloud"),
    )

    # Extract rotations and translations for eye and hand poses
    eye_rots = eye2base_poses[:, :3, :3]
    eye_trans = eye2base_poses[:, :3, 3]
    hand_rots = hand2base_poses[:, :3, :3]
    hand_trans = hand2base_poses[:, :3, 3]

    # Plot camera (eye) trajectory in black
    fig.add_trace(
        go.Scatter3d(
            x=eye_trans[:, 0],
            y=eye_trans[:, 1],
            z=eye_trans[:, 2],
            mode="lines+markers",
            line=dict(color="black", width=2),
            marker=dict(size=3),
            name="Camera Trajectory",
        ),
        row=1,
        col=1,
    )

    # Plot hand trajectory in orange
    fig.add_trace(
        go.Scatter3d(
            x=hand_trans[:, 0],
            y=hand_trans[:, 1],
            z=hand_trans[:, 2],
            mode="lines+markers",
            line=dict(color="orange", width=2),
            marker=dict(size=3),
            name="Hand Trajectory",
        ),
        row=1,
        col=1,
    )

    # Define the length of the coordinate axes for visualization
    axis_length = 0.03  # Adjust based on your scene's scale

    # Plot camera orientation axes (red: x, green: y, blue: z)
    x_axis_x, x_axis_y, x_axis_z = [], [], []
    y_axis_x, y_axis_y, y_axis_z = [], [], []
    z_axis_x, z_axis_y, z_axis_z = [], [], []
    for pos, rot in zip(eye_trans, eye_rots):
        start = pos
        # x-axis (red)
        end = pos + axis_length * rot[:, 0]
        x_axis_x.extend([start[0], end[0], None])
        x_axis_y.extend([start[1], end[1], None])
        x_axis_z.extend([start[2], end[2], None])
        # y-axis (green)
        end = pos + axis_length * rot[:, 1]
        y_axis_x.extend([start[0], end[0], None])
        y_axis_y.extend([start[1], end[1], None])
        y_axis_z.extend([start[2], end[2], None])
        # z-axis (blue)
        end = pos + axis_length * rot[:, 2]
        z_axis_x.extend([start[0], end[0], None])
        z_axis_y.extend([start[1], end[1], None])
        z_axis_z.extend([start[2], end[2], None])

    fig.add_trace(
        go.Scatter3d(
            x=x_axis_x,
            y=x_axis_y,
            z=x_axis_z,
            mode="lines",
            line=dict(color="red", width=2),
            name="Camera X-axis",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=y_axis_x,
            y=y_axis_y,
            z=y_axis_z,
            mode="lines",
            line=dict(color="green", width=2),
            name="Camera Y-axis",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=z_axis_x,
            y=z_axis_y,
            z=z_axis_z,
            mode="lines",
            line=dict(color="blue", width=2),
            name="Camera Z-axis",
        ),
        row=1,
        col=1,
    )

    # Plot hand orientation axes (different colors to distinguish)
    hand_x_axis_x, hand_x_axis_y, hand_x_axis_z = [], [], []
    hand_y_axis_x, hand_y_axis_y, hand_y_axis_z = [], [], []
    hand_z_axis_x, hand_z_axis_y, hand_z_axis_z = [], [], []
    for pos, rot in zip(hand_trans, hand_rots):
        start = pos
        # Hand x-axis (magenta)
        end = pos + axis_length * rot[:, 0]
        hand_x_axis_x.extend([start[0], end[0], None])
        hand_x_axis_y.extend([start[1], end[1], None])
        hand_x_axis_z.extend([start[2], end[2], None])
        # Hand y-axis (orange)
        end = pos + axis_length * rot[:, 1]
        hand_y_axis_x.extend([start[0], end[0], None])
        hand_y_axis_y.extend([start[1], end[1], None])
        hand_y_axis_z.extend([start[2], end[2], None])
        # Hand z-axis (cyan)
        end = pos + axis_length * rot[:, 2]
        hand_z_axis_x.extend([start[0], end[0], None])
        hand_z_axis_y.extend([start[1], end[1], None])
        hand_z_axis_z.extend([start[2], end[2], None])

    fig.add_trace(
        go.Scatter3d(
            x=hand_x_axis_x,
            y=hand_x_axis_y,
            z=hand_x_axis_z,
            mode="lines",
            line=dict(color="magenta", width=2),
            name="Hand X-axis",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=hand_y_axis_x,
            y=hand_y_axis_y,
            z=hand_y_axis_z,
            mode="lines",
            line=dict(color="orange", width=2),
            name="Hand Y-axis",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter3d(
            x=hand_z_axis_x,
            y=hand_z_axis_y,
            z=hand_z_axis_z,
            mode="lines",
            line=dict(color="cyan", width=2),
            name="Hand Z-axis",
        ),
        row=1,
        col=1,
    )

    # Plot the point cloud if provided
    if point_cloud is not None:
        # Automatically detect color format if no colors provided
        if point_colors is None:
            # If the point cloud is in Nx6 format (x, y, z, r, g, b)
            if point_cloud.shape[1] == 6:
                colors = point_cloud[:, 3:]
                point_cloud = point_cloud[:, :3]
                # Normalize colors to range 0-1 (assuming input is 0-255)
                if colors.max() > 1.0:
                    colors = colors / 255.0
                point_colors = [f"rgb({r},{g},{b})" for r, g, b in colors]
            else:
                point_colors = "rgba(100,100,100,0.2)"  # Default gray color

        fig.add_trace(
            go.Scatter3d(
                x=point_cloud[:, 0],
                y=point_cloud[:, 1],
                z=point_cloud[:, 2],
                mode="markers",
                marker=dict(
                    size=1.5,
                    color=point_colors,  # Pass color array or single color
                    opacity=0.8,
                ),
                name="Colored Point Cloud",
            ),
            row=1,
            col=1,
        )

    # Update layout properties
    fig.update_layout(
        title="Hand-Eye Calibration Visualization",
        scene=dict(
            aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"
        ),
        width=1400,
        height=700,
    )

    # Save the result as an HTML file
    fig.write_html(output_path)
    print(f"Visualization saved to: {os.path.abspath(output_path)}")
    return fig


def serve_html(file_path, port=8800):
    """
    Serve an HTML file on the specified port.

    Parameters:
    - file_path: Path to the HTML file to be served.
    - port: Port number on which the server will run (default is 8800).
    """
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import os

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args, directory=os.path.dirname(os.path.abspath(file_path)), **kwargs
            )

    server_address = ("", port)
    httpd = HTTPServer(server_address, Handler)

    print(f"Serving at http://localhost:{port}/{os.path.basename(file_path)}")
    print(
        f"To access remotely, use: http://your-server-ip:{port}/{os.path.basename(file_path)}"
    )
    print("Press Ctrl+C to stop the server")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped")


if __name__ == "__main__":
    pass