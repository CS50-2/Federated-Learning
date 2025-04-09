import base64
import io
import dash
from dash import dcc, html, Input, Output, State, dash_table, MATCH, ALL
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("CSV File Upload and Dynamic Multi-Line Chart"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and drop or click to select a CSV file']),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        accept='.csv'
    ),
    # Store for the parsed DataFrame JSON and current number of line selectors
    dcc.Store(id='stored-data'),
    dcc.Store(id='line-count', data=1),  # Initially, only one line selector is available

    # Container for displaying the upload contents (data preview, dynamic options, charts, and x-range selection)
    html.Div(id='upload-content')
])


def parse_contents(contents, filename):
    """Parse the uploaded file contents into a Pandas DataFrame."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None, html.Div(['Only CSV files are supported.'])
    except Exception as e:
        return None, html.Div(['There was an error processing the file: ' + str(e)])
    return df, None


# Callback: Process file upload, display data preview, dynamic line options, custom chart, and x-axis range selection
@app.callback(
    [Output('upload-content', 'children'),
     Output('stored-data', 'data')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_upload(contents, filename):
    if contents is None:
        return html.Div(["Please upload a CSV file to begin."]), None

    df, error = parse_contents(contents, filename)
    if error is not None:
        return error, None

    # Data preview table (first 5 rows)
    preview_table = dash_table.DataTable(
        id='data-preview',
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.head().to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'}
    )

    # Dynamic line options block
    selectors_block = html.Div([
        html.H2("Line Chart Options"),
        # "Add Line" button
        html.Button("Add Line", id="add-line-btn", n_clicks=0),
        # Container to hold dynamic line selectors
        html.Div(id="line-selectors-container")
    ])

    # x-axis range selector for detailed view
    if "Round" in df.columns:
        x_col = "Round"
    elif pd.api.types.is_numeric_dtype(df[df.columns[0]]):
        x_col = df.columns[0]
    else:
        x_col = None

    if x_col is not None:
        x_lower_default = df[x_col].min()
        x_upper_default = df[x_col].max()
        range_block = html.Div([
            html.H2("Select X-Axis Range (Detailed View)"),
            html.Div([
                html.Div([
                    html.Label("Lower bound: "),
                    dcc.Input(id="x-range-lower", type="number", value=x_lower_default)
                ], style={'display': 'inline-block', 'margin-right': '20px'}),
                html.Div([
                    html.Label("Upper bound: "),
                    dcc.Input(id="x-range-upper", type="number", value=x_upper_default)
                ], style={'display': 'inline-block'})
            ]),
            # New update button for the filtered chart
            html.Button("Update Filtered Chart", id="update-filtered-chart-btn", n_clicks=0, style={'margin-top': '10px'}),
            dcc.Graph(id="filtered-chart", style={'margin-top': '20px'})
        ])
    else:
        # If no valid numeric column is available for x-axis, do not display the range selector.
        range_block = html.Div(["The uploaded data does not contain a numeric column to use as x-axis filtering."])

    # Combine all content together
    content = html.Div([
        html.H2("Data Preview (First 5 Rows)"),
        preview_table,
        selectors_block,
        dcc.Graph(id="custom-chart"),
        html.Div(id="click-output", style={'margin-top': '20px', 'font-size': '20px'}),
        html.Hr(),
        range_block
    ])

    # Store the DataFrame as JSON using 'split' orientation
    data_json = df.to_json(date_format='iso', orient='split')
    return content, data_json


# Callback: Increase the number of line selectors when the "Add Line" button is clicked (max 5)
@app.callback(
    Output('line-count', 'data'),
    Input('add-line-btn', 'n_clicks'),
    State('line-count', 'data')
)
def update_line_count(n_clicks, current_count):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    # Increase the count by one per click, but limit it to 5
    new_count = min(current_count + n_clicks, 5)
    return new_count


# Callback: Dynamically generate line selector panels based on the line count and stored data
@app.callback(
    Output('line-selectors-container', 'children'),
    [Input('line-count', 'data'),
     Input('stored-data', 'data')]
)
def update_line_selectors(line_count, data_json):
    if data_json is None:
        return []

    df = pd.read_json(data_json, orient='split')
    columns = df.columns.tolist()
    dropdown_options = [{"label": col, "value": col} for col in columns]

    selectors = []
    # Create one panel per line selector
    for i in range(line_count):
        default_enable = ['show'] if i == 0 else []  # Only the first is enabled by default
        default_x = columns[0] if len(columns) > 0 else None
        default_y = columns[1] if len(columns) > 1 else None

        selectors.append(
            html.Div([
                html.H4(f"Line {i+1} Options"),
                dcc.Checklist(
                    id={'type': 'line-enable', 'index': i},
                    options=[{'label': 'Display this line', 'value': 'show'}],
                    value=default_enable,
                    labelStyle={'display': 'inline-block'}
                ),
                html.Br(),
                html.Label("X-axis:"),
                dcc.Dropdown(
                    id={'type': 'line-x', 'index': i},
                    options=dropdown_options,
                    value=default_x
                ),
                html.Label("Y-axis:"),
                dcc.Dropdown(
                    id={'type': 'line-y', 'index': i},
                    options=dropdown_options,
                    value=default_y
                ),
                html.Hr()
            ], style={'border': '1px solid #ccc', 'padding': '10px', 'margin-bottom': '10px'})
        )
    return selectors


# Callback: Update the custom chart using all dynamic line selector values
@app.callback(
    Output("custom-chart", "figure"),
    [Input({'type': 'line-enable', 'index': ALL}, 'value'),
     Input({'type': 'line-x', 'index': ALL}, 'value'),
     Input({'type': 'line-y', 'index': ALL}, 'value'),
     Input("stored-data", "data")]
)
def update_chart(line_enables, line_xs, line_ys, data_json):
    if data_json is None:
        return {}
    df = pd.read_json(data_json, orient='split')
    fig = go.Figure()

    # Add a trace for each enabled line if both x and y selections are made
    for i, (enable, x_col, y_col) in enumerate(zip(line_enables, line_xs, line_ys)):
        if (enable is not None and 'show' in enable) and x_col and y_col:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='lines',
                name=f"Line {i+1}: {x_col} vs {y_col}"
            ))
    fig.update_layout(title="Custom Multi-Line Chart",
                      xaxis_title="X-axis",
                      yaxis_title="Y-axis")
    return fig


# Callback: Display details when a point in the custom chart is clicked
@app.callback(
    Output("click-output", "children"),
    [Input("custom-chart", "clickData"),
     Input("stored-data", "data")]
)
def display_click_data(clickData, data_json):
    if data_json is None:
        return "No data uploaded yet."
    df = pd.read_json(data_json, orient='split')
    if clickData is None:
        return "Click on a point in the chart to see details."
    
    point = clickData["points"][0]
    x_value = point["x"]
    y_value = point["y"]
    trace_index = point["curveNumber"]
    
    return f"Clicked point: x = {x_value}, y = {y_value}, Trace index = {trace_index}"


# Callback: Update the filtered chart based on the x-axis range selection and all active dynamic line selections
# This callback is triggered by the "Update Filtered Chart" button.
@app.callback(
    Output("filtered-chart", "figure"),
    Input("update-filtered-chart-btn", "n_clicks"),
    State("x-range-lower", "value"),
    State("x-range-upper", "value"),
    State("stored-data", "data"),
    State({'type': 'line-enable', 'index': ALL}, 'value'),
    State({'type': 'line-x', 'index': ALL}, 'value'),
    State({'type': 'line-y', 'index': ALL}, 'value')
)
def update_filtered_chart(n_clicks, x_lower, x_upper, data_json, line_enables, line_xs, line_ys):
    # If required inputs are missing, return empty figure.
    if n_clicks is None or data_json is None or x_lower is None or x_upper is None:
        return {}
    
    df = pd.read_json(data_json, orient='split')
    fig = go.Figure()
    
    # Loop over all dynamic line selectors; add a trace for each active (enabled) pair.
    any_line_added = False
    for enable, x_col, y_col in zip(line_enables, line_xs, line_ys):
        if enable is not None and 'show' in enable and x_col and y_col:
            # Filter the data for the corresponding x column using the selected range.
            filtered_df = df[(df[x_col] >= x_lower) & (df[x_col] <= x_upper)]
            if not filtered_df.empty:
                fig.add_trace(go.Scatter(
                    x=filtered_df[x_col],
                    y=filtered_df[y_col],
                    mode='lines',
                    name=f"{x_col} vs {y_col}"
                ))
                any_line_added = True
    
    # If no traces were added, update the layout with an informational title.
    if not any_line_added:
        fig.update_layout(title="No data available for the selected range.",
                          xaxis_title="X-axis",
                          yaxis_title="Y-axis")
    else:
        fig.update_layout(title=f"Data from {x_lower} to {x_upper}",
                          xaxis_title="X-axis",
                          yaxis_title="Y-axis")
    
    return fig



if __name__ == '__main__':
    app.run_server(debug=True)
