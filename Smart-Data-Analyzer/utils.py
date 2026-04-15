# utils.py - Utility functions and visualization helpers
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import re
import warnings
warnings.filterwarnings('ignore')

# ── Plotly Theme Configuration ─────────────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(17,24,39,0)",
    plot_bgcolor="rgba(17,24,39,0)",
    font=dict(family="DM Sans", color="#94a3b8"),
    margin=dict(t=48, b=32, l=32, r=24),
    height=420,
)

def apply_theme(fig, height=None):
    """Apply consistent dark theme to plotly figures"""
    kw = dict(CHART_LAYOUT)
    if height:
        kw["height"] = height
    fig.update_layout(**kw)
    fig.update_xaxes(gridcolor="#1e2d45", zerolinecolor="#1e2d45")
    fig.update_yaxes(gridcolor="#1e2d45", zerolinecolor="#1e2d45")
    return fig

# ── UI Components ─────────────────────────────────────────────────────────────
def section(icon, title):
    """Render a styled section header"""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">{icon}</div>
        <h3>{title}</h3>
    </div>""", unsafe_allow_html=True)

def kpi_grid(data: dict, cols=4):
    """Render KPI cards in a grid"""
    items = list(data.items())
    cols_ui = st.columns(min(cols, len(items)))
    
    for i, (label, value) in enumerate(items[:cols * 2]):
        with cols_ui[i % len(cols_ui)]:
            formatted = (f"{value:,.2f}" if isinstance(value, float) else
                        f"{value:,}" if isinstance(value, int) else str(value))
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{formatted}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

def box(text, kind="blue"):
    """Render a styled info box"""
    st.markdown(f'<div class="box box-{kind}">{text}</div>', unsafe_allow_html=True)

def prog_bar(label, pct, color=None):
    """Render a progress bar with label"""
    if color == "green":
        fill = "background:linear-gradient(90deg,#10b981,#06b6d4);"
    elif color == "amber":
        fill = "background:#f59e0b;"
    else:
        fill = "background:linear-gradient(90deg,#3b82f6,#8b5cf6);"
    
    st.markdown(f"""
    <div class="prog-label"><span>{label}</span><span>{pct}%</span></div>
    <div class="prog-wrap"><div class="prog-fill" style="width:{pct}%;{fill}"></div></div>
    """, unsafe_allow_html=True)

def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <div class="header-pill"><span class="pill-dot"></span>LIVE PLATFORM</div>
        <h2>📊 Advanced Data Analytics Platform</h2>
        <p class="tagline">Professional Data Cleaning · Analysis · Visualization · Insights</p>
    </div>""", unsafe_allow_html=True)

def render_empty_state():
    """Render empty state when no data is loaded"""
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;
                justify-content:center;padding:6rem 2rem;text-align:center;">
        <div style="font-size:4rem;margin-bottom:1rem;">🔮</div>
        <h2 style="color:#e2e8f0;font-family:'Syne',sans-serif;">Upload a dataset to begin</h2>
        <p style="color:#475569;max-width:360px;margin-top:.5rem;">
            Drop a CSV or Excel file in the sidebar to unlock the full analytics platform.
        </p>
    </div>""", unsafe_allow_html=True)

# ── Data Processing Utilities ─────────────────────────────────────────────────
def detect_anomalies(df, sample_size=50000):
    """
    Detect anomalies efficiently using sampling for large datasets
    """
    # Use sample for large datasets
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df
        
    out = {}
    for col in sample_df.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = sample_df[col].quantile(.25), sample_df[col].quantile(.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        bad = sample_df[(sample_df[col] < lo) | (sample_df[col] > hi)]
        
        if len(bad):
            # Estimate total outliers in full dataset
            estimated_count = int(len(bad) * (len(df) / len(sample_df)))
            out[col] = {
                "count": estimated_count,
                "percentage": round(len(bad) / len(sample_df) * 100, 1),
                "lower_bound": lo,
                "upper_bound": hi
            }
    return out

def get_recommendations(df):
    """
    Generate actionable recommendations based on data characteristics
    """
    recs = []
    
    # Categorical recommendations
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 5:
        recs.append("💡 Consider encoding categorical variables for ML-ready analysis")
    
    # Missing values
    missing = df.isnull().sum().sum()
    if missing:
        recs.append(f"⚠️ {missing:,} missing values found — run the cleaning step to auto-impute")
    
    # Date columns
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if not len(date_cols):
        recs.append("📅 No date columns detected — time-series analysis unavailable")
    
    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) >= 3:
        recs.append("📊 Multiple numeric columns available — explore the correlation heatmap")
    
    # Dataset size
    if len(df) > 100_000:
        recs.append("💡 Large dataset — apply filters to keep charts responsive")
    
    # Memory usage
    mb = df.memory_usage(deep=True).sum() / 1024**2
    if mb > 100:
        recs.append(f"⚡ High memory usage ({mb:.1f} MB) — consider downcasting column dtypes")
    
    return recs

# ── Chart Creation Functions ─────────────────────────────────────────────────
def create_basic_chart(df, chart_type, x_axis, y_axis=None, color_col=None):
    """Create basic charts with memory efficiency"""
    # Sample large datasets
    if len(df) > 10000:
        chart_df = df.sample(n=min(10000, len(df)), random_state=42)
    else:
        chart_df = df
        
    fig = None
    
    try:
        if chart_type == "Bar Chart":
            fig = px.bar(chart_df, x=x_axis, y=y_axis, 
                        title=f"{y_axis} by {x_axis}" if y_axis else f"Bar Chart: {x_axis}")
        elif chart_type == "Line Chart":
            fig = px.line(chart_df, x=x_axis, y=y_axis,
                         title=f"{y_axis} over {x_axis}" if y_axis else f"Line Chart: {x_axis}")
        elif chart_type == "Scatter Plot":
            fig = px.scatter(chart_df, x=x_axis, y=y_axis, color=color_col,
                           title=f"{y_axis} vs {x_axis}" if y_axis else f"Scatter: {x_axis}")
        elif chart_type == "Histogram":
            fig = px.histogram(chart_df, x=x_axis, nbins=30,
                             title=f"Distribution of {x_axis}")
        elif chart_type == "Box Plot":
            if y_axis:
                fig = px.box(chart_df, x=x_axis if x_axis else None, y=y_axis,
                           title=f"Distribution of {y_axis}" if not x_axis else f"{y_axis} by {x_axis}")
            else:
                fig = px.box(chart_df, y=y_axis, title=f"Distribution of {y_axis}")
    except Exception as e:
        return None, str(e)
        
    return fig, None

def create_advanced_chart(df, chart_type):
    """Create advanced chart types"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Sample for performance
    if len(df) > 5000:
        chart_df = df.sample(n=5000, random_state=42)
    else:
        chart_df = df
        
    fig = None
    
    try:
        if chart_type == "3D Scatter Plot" and len(num_cols) >= 3:
            fig = px.scatter_3d(chart_df,
                                x=num_cols[0], y=num_cols[1], z=num_cols[2],
                                color=num_cols[2], color_continuous_scale="Viridis",
                                title="3D Scatter")
        elif chart_type == "Sunburst Chart" and len(cat_cols) >= 2:
            fig = px.sunburst(chart_df, path=cat_cols[:3], title="Sunburst")
        elif chart_type == "Treemap" and cat_cols and num_cols:
            fig = px.treemap(chart_df, path=[cat_cols[0]], values=num_cols[0],
                            color=num_cols[0], color_continuous_scale="Blues",
                            title="Treemap")
        elif chart_type == "Parallel Coordinates" and len(num_cols) >= 2:
            fig = px.parallel_coordinates(chart_df.head(500), dimensions=num_cols[:6],
                                          color=num_cols[0], color_continuous_scale="Viridis",
                                          title="Parallel Coordinates")
    except Exception as e:
        return None, str(e)
        
    return fig, None

# ── Export Functions ─────────────────────────────────────────────────────────
def export_to_excel(df, cleaning_log):
    """Export cleaned data to Excel with multiple sheets"""
    output = io.BytesIO()
    num_df = df.select_dtypes(include=[np.number])
    
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Cleaned data
        df.to_excel(writer, sheet_name="Cleaned Data", index=False)
        
        # Statistics
        if not num_df.empty:
            num_df.describe().round(2).to_excel(writer, sheet_name="Statistics")
        
        # Data quality
        pd.DataFrame({
            "Metric": ["Total Records", "Total Columns", "Missing Values", "Cleaning Actions"],
            "Value": [len(df), len(df.columns), int(df.isnull().sum().sum()), len(cleaning_log)]
        }).to_excel(writer, sheet_name="Data Quality", index=False)
        
        # Column info
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Dtype": df.dtypes.astype(str).values,
            "Unique": [df[c].nunique() for c in df.columns],
            "Nulls": df.isnull().sum().values
        })
        col_info.to_excel(writer, sheet_name="Column Info", index=False)
        
    output.seek(0)
    return output

def export_to_html(df, title="DataSphere Export"):
    """Export data to HTML table"""
    tbl = df.head(1000).to_html(classes="table", index=False, max_rows=1000)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #0a0e1a; color: #e2e8f0; }}
        h1 {{ color: #3b82f6; font-family: system-ui; }}
        .table {{ width: 100%; border-collapse: collapse; }}
        .table th, .table td {{ border: 1px solid #1e2d45; padding: 8px 12px; text-align: left; }}
        .table th {{ background: #111827; color: #94a3b8; font-size: .8rem; text-transform: uppercase; }}
        .table tr:nth-child(even) {{ background: #111827; }}
        .table tr:hover {{ background: #1a2235; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p style="color:#475569">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · {len(df):,} records (showing first 1000)</p>
    {tbl}
</body>
</html>"""
    
    return html

# ── Session State Management ─────────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "original_df": None,
        "cleaned_df": None,
        "cleaning_log": [],
        "data_cleaned": False,
        "show_analysis": False,
        "file_name": "",
        "analyzer": None,
        "cleaner": None
    }
    
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def reset_session():
    """Reset session state for new dataset"""
    st.session_state.original_df = None
    st.session_state.cleaned_df = None
    st.session_state.cleaning_log = []
    st.session_state.data_cleaned = False
    st.session_state.show_analysis = False
    st.session_state.file_name = ""
    st.session_state.analyzer = None
    st.session_state.cleaner = None

# ── Validation Functions ─────────────────────────────────────────────────────
def validate_range(df, column, min_val, max_val):
    """Validate numeric range"""
    if column not in df.columns:
        return None, "Column not found"
    
    if df[column].dtype not in [np.number]:
        return None, "Column must be numeric"
    
    invalid = df[(df[column] < min_val) | (df[column] > max_val)]
    return invalid, None

def validate_pattern(df, column, pattern):
    """Validate regex pattern"""
    if column not in df.columns:
        return None, "Column not found"
    
    try:
        matches = df[column].astype(str).str.match(pattern)
        invalid = df[~matches]
        return invalid, None
    except re.error as e:
        return None, f"Invalid regex: {e}"

def validate_required(df, column):
    """Validate required field (no nulls)"""
    if column not in df.columns:
        return None, "Column not found"
    
    missing = df[column].isnull().sum()
    return missing, None