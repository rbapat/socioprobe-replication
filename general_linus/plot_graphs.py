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
        Dataframe of all combined datasets
        Has columns "model", "score_type", "layer", "run1", "run2",
            "run3", "run4", "run5", "mean", "std_dev"
    metric : str
        metric used for y-axis, either "F1" or "MDL", from "score_type" column
    layer : int, optional
        layer to probe and plot, by default 12 (should be last layer)
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
        error_y=dict(type='data', array=final_embed.loc[final_embed["model"] == "DeBERTa-base", "std_dev"])
    ))

    # Customization
    fig.update_layout(legend=dict(yanchor="top",
                                  y=0.35,
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


def plot_all_layers(dataset: pd.DataFrame, data_name: str, metric: str):
    """Plot metric vs. layer, given a dataset from the probes

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset containing accuracies for each probe
        Must have columns "mean", "std_dev", "model", "score_type", "data_name",
            and "layer"
    data_name : str
        Name of dataset, must be found within "data_name" column
    metric : str
        Metric of interest within "score_type" of column
        either "F1" or "MDL"
    """
    df = dataset.copy()

    df["upper"] = df["mean"] + df["std_dev"]
    df["lower"] = df["mean"] - df["std_dev"]
    electra_df = (df.loc[
        (df["model"] == "ELECTRA") &
        (df["score_type"] == metric) &
        (df["data_name"] == data_name), :]
                  ).sort_values(by=["layer"])
    roberta_df = (df.loc[
        (df["model"] == "RoBERTa-base") &
        (df["score_type"] == metric) &
        (df["data_name"] == data_name), :]
                  ).sort_values(by=["layer"])
    deberta_df = (df.loc[
        (df["model"] == "DeBERTa-base") &
        (df["score_type"] == metric) &
        (df["data_name"] == data_name), :]
                  ).sort_values(by=["layer"])

    fig = go.Figure([
        # Bounds for ELECTRA
        go.Scatter(
            name='Upper Bound',
            x=electra_df["layer"],
            y=electra_df["upper"],
            mode='lines',
            marker=dict(color='rgba(86, 225, 232, 0.4)'),
            line=dict(width=0),
            showlegend=False,
            opacity=0.2
        ),
        go.Scatter(
            name='Lower Bound',
            x=electra_df["layer"],
            y=electra_df["lower"],
            marker=dict(color='rgba(86, 225, 232, 0.4)'),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(86, 225, 232, 0.4)',
            fill='tonexty',
            showlegend=False,
            opacity=0.2
        ),
        # Bounds for RoBERTa-base
        go.Scatter(
            name='Upper Bound',
            x=roberta_df["layer"],
            y=roberta_df["upper"],
            mode='lines',
            marker=dict(color='rgba(110, 240, 112, 0.4)'),
            line=dict(width=0),
            showlegend=False,
            opacity=0.2
        ),
        go.Scatter(
            name='Lower Bound',
            x=roberta_df["layer"],
            y=roberta_df["lower"],
            marker=dict(color='rgba(110, 240, 112, 0.4)'),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(110, 240, 112, 0.4)',
            fill='tonexty',
            showlegend=False,
            opacity=0.2
        ),
        # Bounds for DeBERTa-base
        go.Scatter(
            name='Upper Bound',
            x=deberta_df["layer"],
            y=deberta_df["upper"],
            mode='lines',
            marker=dict(color='rgba(203, 110, 240, 0.4)'),
            line=dict(width=0),
            showlegend=False,
            opacity=0.2
        ),
        go.Scatter(
            name='Lower Bound',
            x=deberta_df["layer"],
            y=deberta_df["lower"],
            marker=dict(color='rgba(203, 110, 240, 0.4)'),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(203, 110, 240, 0.4)',
            fill='tonexty',
            showlegend=False,
            opacity=0.2
        ),

        # Graph lines
        go.Scatter(
            name="DeBERTa-base",
            x=deberta_df["layer"],
            y=deberta_df["mean"],
            line=dict(color="purple"),
            mode='lines'
        ),
        go.Scatter(
            name="RoBERTa-base",
            x=roberta_df["layer"],
            y=roberta_df["mean"],
            line=dict(color="green"),
            mode='lines'
        ),
        go.Scatter(
            name="ELECTRA",
            x=electra_df["layer"],
            y=electra_df["mean"],
            line=dict(color="royalblue"),
            mode='lines'
        ),
    ])

    # Add customizations
    fig.update_layout(title=f"Layer-wise {metric} Scores for {data_name}")
    fig.update_xaxes(title="Layer")
    fig.update_yaxes(title=f"Test {metric} score")
    fig.update_layout(legend=dict(yanchor="top",
                                  xanchor="right",
                                  x=0.99,
                                  bgcolor='rgba(169, 163, 172, 0.3)'
                                  ),
                      legend_title="Model")

    # Save and show model
    os.makedirs("figures", exist_ok=True)
    path = os.path.join("figures", f"RQ2_{data_name}_{metric}.png")
    fig.write_image(path)
    fig.show()


if __name__ == "__main__":
    # VARIABLES - TODO: Add datasets here
    cola_scores = pd.read_csv("scores/cola_scores.csv", index_col=0)
    nps_scores = pd.read_csv("scores/NPS_scores.csv", index_col=0)
    mdgender_scores = pd.read_csv("scores/mdgender_scores.csv", index_col=0)
    joc_electra_scores = pd.read_csv("scores/google_electra-base-discriminator_results.csv", index_col=0)
    joc_deberta_scores = pd.read_csv("scores/microsoft_deberta-base_results.csv", index_col=0)
    joc_roberta_scores = pd.read_csv("scores/roberta-base_results.csv", index_col=0)

    # Join dfs and find means and averages for each row
    # TODO: UPDATE LIST
    df_joined = pd.concat([cola_scores, nps_scores, mdgender_scores, joc_electra_scores,
                           joc_deberta_scores, joc_roberta_scores], ignore_index=True)
    df_joined["mean"] = df_joined[["run1", "run2", "run3", "run4", "run5"]].mean(axis=1)
    df_joined["std_dev"] = df_joined[["run1", "run2", "run3", "run4", "run5"]].std(axis=1)

    # ~ RQ1: Final layer embeddings
    for metric in ["F1", "MDL"]:
        plot_final_layer(df_joined, metric)

    # ~ RQ2: Graphs for each dataset
    for data_name in df_joined["data_name"].unique():
        for score_type in df_joined["score_type"].unique():
            plot_all_layers(df_joined, data_name, score_type)