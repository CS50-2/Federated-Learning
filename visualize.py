import dash
from dash import dcc, html, Output, Input
import pandas as pd
import plotly.express as px

# Load the CSV data (assume your CSV has columns: Round, Accuracy_fedGRA, Accuracy_high_loss, Accuracy_fedavg)
df = pd.read_csv("comparison_results.csv")

# Convert the wide format to long format for easier plotting
df_long = df.melt(id_vars="Round", value_vars=["Accuracy_fedGRA", "Accuracy_high_loss", "Accuracy_fedavg"],
                  var_name="Method", value_name="Accuracy")

# Create the main line plot using Plotly Express
fig = px.line(df_long, x="Round", y="Accuracy", color="Method",
              title="Comparison of FedGRA, High-Loss Filtering, and FedAvg")

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Federated Learning Experiment Visualization"),
    dcc.Graph(id="line-chart", figure=fig),
    html.Div(id="click-output", style={'margin-top': '20px', 'font-size': '20px'}),
    
    html.H2("Select x-range for detailed view"),
    html.Div([
        html.Label("Lower bound: "),
        dcc.Input(id="x-range-lower", type="number", value=df["Round"].min())
    ], style={'display': 'inline-block', 'margin-right': '20px'}),
    html.Div([
        html.Label("Upper bound: "),
        dcc.Input(id="x-range-upper", type="number", value=df["Round"].max())
    ], style={'display': 'inline-block'}),
    
    dcc.Graph(id="filtered-chart", style={'margin-top': '20px'})
])

# Callback to capture click events on the main graph
@app.callback(
    Output("click-output", "children"),
    Input("line-chart", "clickData")
)
def display_click_data(clickData):
    if clickData is None:
        return "Click on a point in the graph to see the accuracies for each method."
    
    # Extract the x value from the clickData (which is the Round number)
    x_value = clickData["points"][0]["x"]
    
    # Retrieve the corresponding row from the DataFrame
    filtered = df[df["Round"] == x_value]
    if filtered.empty:
        return f"No data found for round {x_value}."
    
    acc_fedGRA = filtered["Accuracy_fedGRA"].values[0]
    acc_high_loss = filtered["Accuracy_high_loss"].values[0]
    acc_fedavg = filtered["Accuracy_fedavg"].values[0]
    
    return (
        f"Round: {x_value} | "
        f"FedGRA Accuracy: {acc_fedGRA:.2f}% | "
        f"High-Loss Accuracy: {acc_high_loss:.2f}% | "
        f"FedAvg Accuracy: {acc_fedavg:.2f}%"
    )

# New callback for updating filtered chart based on x-range selection
@app.callback(
    Output("filtered-chart", "figure"),
    [Input("x-range-lower", "value"),
     Input("x-range-upper", "value")]
)
def update_filtered_chart(x_lower, x_upper):
    # Check for valid inputs
    if x_lower is None or x_upper is None or x_lower > x_upper:
        return {}  # Return empty figure if inputs are invalid
    
    # Filter the data based on the selected x-range
    filtered_df = df[(df["Round"] >= x_lower) & (df["Round"] <= x_upper)]
    if filtered_df.empty:
        return {}  # Return empty figure if no data is found in the selected range

    # Convert the filtered data to long format for plotting
    df_long_filtered = filtered_df.melt(id_vars="Round", value_vars=["Accuracy_fedGRA", "Accuracy_high_loss", "Accuracy_fedavg"],
                                        var_name="Method", value_name="Accuracy")
    
    # Create the filtered line plot
    fig_filtered = px.line(df_long_filtered, x="Round", y="Accuracy", color="Method",
                           title=f"Model Accuracies from Round {x_lower} to {x_upper}")
    return fig_filtered

if __name__ == "__main__":
    app.run_server(debug=True)
