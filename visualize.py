import base64
import io
import dash
from dash import dcc, html, Input, Output, State, dash_table, MATCH, ALL
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# Initialising app 
app = dash.Dash(__name__)


app.layout = html.Div([
    html.H1("CSV File Upload and Dynamic Multi-Line Chart"),
    # csv upload interface
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
    # Store for the uploaded data and the current count of dynamic line selectors
    dcc.Store(id='stored-data'),
    dcc.Store(id='line-count', data=1),  # Initially, only one dynamic line selector

    # Main page content container
    html.Div(id='upload-content')
])

# Parse the uploaded file contents into a Pandas DataFrame.
def parse_contents(contents, filename):

    _, content_string = contents.split(',') # base64,<encoded_content>
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None, html.Div(['Only CSV files are supported.'])
    except Exception as e:
        return None, html.Div(['There was an error processing the file: ' + str(e)])
    return df, None

# Callback: Process file upload, display data preview, dynamic line options, charts, x-axis range filter, and comparison chart inputs.
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

    # Data preview table: show first 5 rows
    preview_table = dash_table.DataTable(
        id='data-preview',
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.head().to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'}
    )

    # Dynamic line chart options block
    selectors_block = html.Div([
        html.H2("Line Chart Options"),
        html.Button("Add Line", id="add-line-btn", n_clicks=0),
        html.Div(id="line-selectors-container")
    ])

    # x-axis range selection block (detailed view)
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
            html.Button("Update Filtered Chart", id="update-filtered-chart-btn", n_clicks=0, style={'margin-top': '10px'}),
            dcc.Graph(id="filtered-chart", style={'margin-top': '20px'})
        ])
    else:
        range_block = html.Div(["The uploaded data does not contain a numeric column to use as x-axis filtering."])

    # Comparison chart section with range selection inputs
    comparison_block = html.Div([
        html.H2("Comparison Chart: Diverging Bar Plot"),
        html.Div([
            html.Label("Group by Column:"),
            dcc.Dropdown(id="groupby-column", options=[], placeholder="Select grouping column")
        ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '20px'}),
        html.Div([
            html.Label("Barplot Column 1:"),
            dcc.Dropdown(id="barplot-col1", options=[], placeholder="Select first numeric column")
        ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '20px'}),
        html.Div([
            html.Label("Barplot Column 2:"),
            dcc.Dropdown(id="barplot-col2", options=[], placeholder="Select second numeric column")
        ], style={'width': '30%', 'display': 'inline-block'}),
        html.Br(),
        # New range selection for the comparison chart (applied on the groupby column if numeric)
        html.Div([
            html.Label("Select Group Range:"),
            html.Div([
                dcc.Input(id="comp-range-lower", type="number", placeholder="Lower bound"),
                dcc.Input(id="comp-range-upper", type="number", placeholder="Upper bound")
            ], style={'display': 'inline-block', 'margin-right': '20px'})
        ], style={'margin-top': '10px'}),
        html.Button("Update Comparison Chart", id="update-comp-chart-btn", n_clicks=0, style={'margin-top': '10px'}),
        dcc.Graph(id="comparison-chart", style={'margin-top': '20px'})
    ], style={'margin-top': '40px'})

    # Combine all components into the main content layout
    content = html.Div([
        html.H2("Data Preview (First 5 Rows)"),
        preview_table,
        selectors_block,
        dcc.Graph(id="custom-chart"),
        html.Div(id="click-output", style={'margin-top': '20px', 'font-size': '20px'}),
        html.Hr(),
        range_block,
        html.Hr(),
        comparison_block
    ])

    # Store the DataFrame as JSON using the 'split' orientation
    data_json = df.to_json(date_format='iso', orient='split')
    return content, data_json

# Callback: Increase the number of dynamic line selectors when "Add Line" is clicked (max 5)
@app.callback(
    Output('line-count', 'data'),
    Input('add-line-btn', 'n_clicks'),
    State('line-count', 'data')
)
def update_line_count(n_clicks, current_count):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    new_count = min(current_count + n_clicks, 5)
    return new_count

# Callback: Update dynamic line selectors based on current count and stored data.
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
    for i in range(line_count):
        default_enable = ['show'] if i == 0 else []
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

# Callback: Update the custom multi-line chart based on dynamic selector inputs.
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

    # Add a trace for each enabled line based on the selected x and y columns
    for i, (enable, x_col, y_col) in enumerate(zip(line_enables, line_xs, line_ys)):
        if (enable is not None and 'show' in enable) and x_col and y_col:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col].rolling(window=10, min_periods=1).mean(), 
                mode='lines',
                name=f"Line {i+1}: {x_col} vs {y_col}"
            ))
    fig.update_layout(title="Custom Multi-Line Chart",
                      xaxis_title="X-axis",
                      yaxis_title="Y-axis")
    return fig

# Callback: When clicking on a point in the custom chart, show detailed information.
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

# Callback: Update the filtered chart based on selected x-axis range and dynamic line selections.
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
    if n_clicks is None or data_json is None or x_lower is None or x_upper is None:
        return {}
    
    df = pd.read_json(data_json, orient='split')
    fig = go.Figure()
    
    any_line_added = False
    for enable, x_col, y_col in zip(line_enables, line_xs, line_ys):
        if enable is not None and 'show' in enable and x_col and y_col:
            filtered_df = df[(df[x_col] >= x_lower) & (df[x_col] <= x_upper)]
            if not filtered_df.empty:
                fig.add_trace(go.Scatter(
                    x=filtered_df[x_col],
                    y=filtered_df[y_col].rolling(window=10, min_periods=1).mean(),
                    mode='lines',
                    name=f"{x_col} vs {y_col}"
                ))
                any_line_added = True
    
    if not any_line_added:
        fig.update_layout(title="No data available for the selected range.",
                          xaxis_title="X-axis",
                          yaxis_title="Y-axis")
    else:
        fig.update_layout(title=f"Data from {x_lower} to {x_upper}",
                          xaxis_title="X-axis",
                          yaxis_title="Y-axis")
    
    return fig

# Callback: When the uploaded data changes, update the dropdown options for the comparison chart.
@app.callback(
    [Output("groupby-column", "options"),
     Output("barplot-col1", "options"),
     Output("barplot-col2", "options")],
    Input("stored-data", "data")
)
def update_dropdown_options(data_json):
    if data_json is None:
        return [], [], []
    df = pd.read_json(data_json, orient='split')
    # Allow all columns for grouping
    groupby_options = [{"label": col, "value": col} for col in df.columns]
    # Only allow numeric columns for barplot selections
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    numeric_options = [{"label": col, "value": col} for col in numeric_cols]
    return groupby_options, numeric_options, numeric_options

# Callback: Update the diverging bar plot (comparison chart) after the user clicks the button.
# A range filter on the groupby column is also applied if provided.
@app.callback(
    Output("comparison-chart", "figure"),
    Input("update-comp-chart-btn", "n_clicks"),
    State("groupby-column", "value"),
    State("barplot-col1", "value"),
    State("barplot-col2", "value"),
    State("stored-data", "data"),
    State("comp-range-lower", "value"),
    State("comp-range-upper", "value")
)
def update_comparison_chart(n_clicks, groupby_col, col1, col2, data_json, comp_lower, comp_upper):
    # Validate input: if button hasn't been clicked or required inputs are missing, return a message figure.
    if n_clicks is None or n_clicks == 0 or not all([groupby_col, col1, col2, data_json]):
        return go.Figure().update_layout(title="Please select Group by and both numeric columns to compare.")
    
    df = pd.read_json(data_json, orient='split')
    
    # Group by the selected column and aggregate only the selected numeric columns
    try:
        grouped_df = df.groupby(groupby_col)[[col1, col2]].mean().reset_index()
    except Exception as e:
        return go.Figure().update_layout(title=f"Error during grouping: {e}")
    
    # If a numeric range is provided, attempt to convert the groupby column to numeric and filter groups
    if comp_lower is not None and comp_upper is not None:
        try:
            grouped_df[groupby_col] = pd.to_numeric(grouped_df[groupby_col], errors='coerce')
            grouped_df = grouped_df[(grouped_df[groupby_col] >= comp_lower) & (grouped_df[groupby_col] <= comp_upper)]
        except Exception as e:
            pass

    # Sort the grouped data to ensure a consistent x-axis order
    grouped_df = grouped_df.sort_values(by=groupby_col)

    trace1 = go.Bar(
        x=grouped_df[groupby_col],
        y=grouped_df[col1],
        base=0,  
        name=col1
    )
    trace2 = go.Bar(
        x=grouped_df[groupby_col],
        y=-grouped_df[col2], 
        base=0,
        name=col2
    )
    
    # Build the figure and add a horizontal reference line at 0
    fig = go.Figure(data=[trace1, trace2])
    fig.add_shape(
        type="line",
        x0=min(grouped_df[groupby_col]),
        x1=max(grouped_df[groupby_col]),
        y0=0,   # Reference line at 0
        y1=0,
        line=dict(color="Black", width=2, dash="dash")
    )

    max_val1 = grouped_df[col1].max()
    max_val2 = grouped_df[col2].max()
    max_val = max(max_val1, max_val2)
    
    tick_vals = [-max_val, -max_val/2, 0, max_val/2, max_val]
    tick_text = [str(round(abs(x), 2)) for x in tick_vals]
    
    fig.update_yaxes(
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_text
    )

    fig.update_layout(
        title=f"Comparison of {col1} and {col2} grouped by {groupby_col}",
        xaxis_title=groupby_col,
        yaxis_title="Value",
        barmode='overlay'  # Overlay mode, display up and down
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
