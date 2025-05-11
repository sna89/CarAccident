import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Union, Optional

def create_line_chart(
    data: List[Dict],
    x_column: str,
    y_column: str,
    group_column: str,
    title: str = "Line Chart",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    group_label: Optional[str] = None,
    markers: bool = True,
    line_width: int = 2,
    marker_size: int = 8,
    height: Optional[int] = None,
    width: Optional[int] = None
) -> go.Figure:
    """
    Create a Plotly line chart with customizable parameters.
    
    Args:
        data (List[Dict]): List of dictionaries containing the data
        x_column (str): Name of the column to use for x-axis (e.g., 'year')
        y_column (str): Name of the column to use for y-axis (e.g., 'accident_count')
        group_column (str): Name of the column to use for grouping (e.g., 'severity')
        title (str): Title of the chart
        x_label (str, optional): Label for x-axis. If None, uses x_column
        y_label (str, optional): Label for y-axis. If None, uses y_column
        group_label (str, optional): Label for the group legend. If None, uses group_column
        markers (bool): Whether to show markers on the lines
        line_width (int): Width of the lines
        marker_size (int): Size of the markers
        height (int, optional): Height of the chart in pixels
        width (int, optional): Width of the chart in pixels
    
    Returns:
        go.Figure: A Plotly figure object
    
    Example:
        data = [
            {'year': 2020, 'severity': 'fatal', 'accident_count': 5},
            {'year': 2020, 'severity': 'serious', 'accident_count': 10},
            {'year': 2021, 'severity': 'fatal', 'accident_count': 7},
            {'year': 2021, 'severity': 'serious', 'accident_count': 12}
        ]
        fig = create_line_chart(
            data=data,
            x_column='year',
            y_column='accident_count',
            group_column='severity',
            title='Accidents Over Time by Severity'
        )
    """
    # Create DataFrame from the data
    df = pd.DataFrame(data)
    
    # Verify required columns exist
    required_columns = [x_column, y_column, group_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Create the line chart
    fig = px.line(
        df,
        x=x_column,
        y=y_column,
        color=group_column,
        title=title,
        markers=markers,
        height=height,
        width=width
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_label or x_column,
        yaxis_title=y_label or y_column,
        showlegend=True,
        legend_title=group_label or group_column,
        hovermode='x unified'
    )
    
    # Update traces
    fig.update_traces(
        line=dict(width=line_width),
        marker=dict(size=marker_size)
    )
    
    return fig 