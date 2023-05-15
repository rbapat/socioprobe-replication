# Goal: Combine all dataset metrics to get a final graph
# There should be graphs relating to each research question
# as well as the dataset and model used

import os
import plotly.graph_objs as go
import pandas as pd


def plot_final_layer(dataset: pd.DataFrame, metric: str, layer: int = 12):
    """Plot the accuracies for the final layer
    Expects a dataframe with "model", "score_type", "layer",
        "run1", "run2", "run3", "run4", "run5", "mean", "std_dev" columns


    Parameters
    ----------
    dataset : pd.DataFrame
        _description_
    metric : str
        _description_
    layer : int, optional
        _description_, by default 12
    """
    final_embed = dataset.loc[(dataset["score_type"] == metric) & (dataset["layer"] == layer), :]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=final_embed.loc[final_embed["model"] == "ELECTRA", "data_name"],
        y=final_embed.loc[final_embed["model"] == "ELECTRA", "mean"],
        name='ELECTRA',
        marker_color='royalblue',
        error_y=dict(type='data', array=final_embed.loc[final_embed["model"] == "ELECTRA", "std_dev"])
    ))

    fig.add_trace(go.Bar(
        x=final_embed.loc[final_embed["model"] == "RoBERTa-base", "data_name"],
        y=final_embed.loc[final_embed["model"] == "RoBERTa-base", "mean"],
        name="RoBERTa-base",
        marker_color='green',
        error_y=dict(type='data', array=final_embed.loc[final_embed["model"] == "RoBERTa-base", "std_dev"])
    ))

    fig.add_trace(go.Bar(
        x=final_embed.loc[final_embed["model"] == "DeBERTa-base", "data_name"],
        y=final_embed.loc[final_embed["model"] == "DeBERTa-base", "mean"],
        name="DeBERTa-base",
        marker_color='purple',
        error_y=dict(type='data', array=final_embed.loc[final_embed["model"] == "RoBERTa-base", "std_dev"])
    ))

    fig.update_layout(legend=dict(yanchor="top",
                                  xanchor="right"
                                  ),
                      legend_title="Model")
    fig.update_yaxes(title=f"Test {metric} Score")

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group', xaxis_tickangle=-45)

    # Save figures
    os.makedirs("figures", exist_ok=True)
    path = os.path.join("figures", f"RQ1_{metric}.png")
    fig.write_image(path)
    fig.show()

if __name__ == "__main__":
    # VARIABLES - TODO: Add datasets here
    cola_scores = pd.read_csv("scores/cola_scores.csv", index_col=0)
    nps_scores = pd.read_csv("scores/NPS_scores.csv", index_col=0)

    # Join data
    df_joined = pd.concat([cola_scores, nps_scores], ignore_index=True)

    # Find means and averages for each row
    df_joined["mean"] = df_joined[["run1", "run2", "run3", "run4", "run5"]].mean(axis=1)
    df_joined["std_dev"] = df_joined[["run1", "run2", "run3", "run4", "run5"]].std(axis=1)


    # ~ RQ1: Final layer embeddings
    for metric in ["F1", "MDL"]:
        plot_final_layer(df_joined, metric)


    # ~ RQ2: Graphs for each dataset