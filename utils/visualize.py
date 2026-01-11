import os

import pandas as pd
import plotly.graph_objects as go


def plot_predictions(y_true, y_pred, save_html_path: str, save_csv: bool = False):
    """Save an interactive Pred vs True scatter as HTML.

    The legacy version always wrote a sidecar CSV next to the HTML. This project
    keeps CSV outputs strictly controlled by the pipeline, so the default is now
    `save_csv=False`.
    """
    out_dir = os.path.dirname(save_html_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame({"True": y_true, "Predicted": y_pred})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="Predictions"))
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_true,
            mode="lines",
            name="y=x",
            line=dict(dash="dash"),
        )
    )
    fig.update_layout(title="Predicted vs True", xaxis_title="True", yaxis_title="Predicted")
    fig.write_html(save_html_path)

    if save_csv:
        df.to_csv(save_html_path.replace(".html", ".csv"), index=False)
