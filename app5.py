import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Content Strategy Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .nav-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        display: flex;
        justify-content: center;
    }
    .feature-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        height: 100%;
    }
    .upload-section {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #2E86AB;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .insight-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ContentAnalyzer:
    def __init__(self):
        self.df = None
    
    def load_data(self, uploaded_file):
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV, Excel, or JSON file.")
                return None
            
            # Convert date column if exists
            date_columns = ['date', 'Date', 'published_date', 'created_at']
            for col in date_columns:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        st.warning(f"Could not convert column '{col}' to datetime")
                    break
            
            return df
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def validate_data(self, df):
        """Validate the uploaded dataset has required columns"""
        st.subheader("🔍 Data Validation")
        
        # Check for required columns
        required_columns = {
            'content_type': ['content_type', 'type', 'contentType', 'category'],
            'views': ['views', 'view_count', 'impressions', 'page_views'],
            'engagement': ['engagement_rate', 'engagement', 'engagementRate', 'click_rate']
        }
        
        column_mapping = {}
        missing_columns = []
        
        for required_col, alternatives in required_columns.items():
            found = False
            for alt in alternatives:
                if alt in df.columns:
                    column_mapping[required_col] = alt
                    found = True
                    break
            if not found:
                missing_columns.append(required_col)
        
        if missing_columns:
            st.warning(f"⚠️ Missing recommended columns: {', '.join(missing_columns)}")
            st.info("""
            **Recommended columns for full functionality:**
            - `content_type` (Blog Post, Social Media, etc.)
            - `views` or `view_count` (numeric)
            - `engagement_rate` or `engagement` (numeric, 0-1 scale)
            - `writing_style` (Professional, Conversational, etc.)
            - `client` (client names)
            - `date` (publication date)
            """)
        else:
            st.success("✅ All recommended columns found!")
        
        # Show data preview
        st.write("**Data Preview:**")
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        return column_mapping
    
    def get_performance_metrics(self, df, column_mapping):
        """Calculate key performance metrics"""
        try:
            # Map columns
            views_col = column_mapping.get('views', 'views')
            engagement_col = column_mapping.get('engagement', 'engagement_rate')
            
            metrics = {
                'Total Content': len(df),
                'Avg Views': f"{df[views_col].mean():.0f}" if views_col in df.columns else 'N/A',
                'Engagement Rate': f"{df[engagement_col].mean() * 100:.2f}%" if engagement_col in df.columns else 'N/A',
            }
            
            # Additional metrics if columns exist
            if 'read_time_minutes' in df.columns:
                metrics['Avg Read Time'] = f"{df['read_time_minutes'].mean():.1f} min"
            if 'shares' in df.columns:
                metrics['Total Shares'] = f"{df['shares'].sum():,}"
            if 'completion_rate' in df.columns:
                metrics['Completion Rate'] = f"{df['completion_rate'].mean() * 100:.1f}%"
                
            return metrics
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return {}

def create_navigation():
    """Create the top navigation tabs"""
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    # Navigation with "Begin" tab instead of "Dashboard"
    nav_options = ["About Tool", "Begin"]
    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        key="nav_radio",
        label_visibility="collapsed",
        horizontal=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    return selected_nav

def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    clients = ['TechCorp', 'HealthPlus', 'EduFuture', 'FinancePro', 'LifestyleCo']
    content_types = ['Blog Post', 'Social Media', 'Newsletter', 'Case Study', 'Whitepaper']
    writing_styles = ['Professional', 'Conversational', 'Technical', 'Storytelling', 'Persuasive']
    
    data = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(100):
        client = np.random.choice(clients)
        content_type = np.random.choice(content_types)
        writing_style = np.random.choice(writing_styles)
        
        # Base performance metrics
        if content_type == 'Blog Post':
            views = np.random.randint(1000, 5000)
            engagement_rate = np.random.uniform(0.02, 0.08)
        elif content_type == 'Social Media':
            views = np.random.randint(5000, 20000)
            engagement_rate = np.random.uniform(0.05, 0.15)
        elif content_type == 'Newsletter':
            views = np.random.randint(2000, 8000)
            engagement_rate = np.random.uniform(0.03, 0.10)
        else:
            views = np.random.randint(500, 3000)
            engagement_rate = np.random.uniform(0.01, 0.05)
        
        # Adjust based on writing style
        if writing_style == 'Conversational':
            engagement_rate *= 1.2
        elif writing_style == 'Storytelling':
            engagement_rate *= 1.15
        
        data.append({
            'date': start_date + timedelta(days=np.random.randint(0, 180)),
            'client': client,
            'content_type': content_type,
            'writing_style': writing_style,
            'title': f'Content Piece {i+1}',
            'views': views,
            'engagement_rate': round(engagement_rate, 3),
            'read_time_minutes': np.random.randint(2, 15),
            'completion_rate': round(np.random.uniform(0.3, 0.9), 2),
            'shares': int(views * engagement_rate * np.random.uniform(0.1, 0.3)),
            'word_count': np.random.randint(500, 2000)
        })
    
    return pd.DataFrame(data)

def display_about_tool():
    """Display the About Tool section"""
    # About This Tool Section
    st.markdown("""
    <div class="feature-card">
        <h3>About This AI Tool</h3>
        <p>This AI-powered tool is designed to assist content creators, marketers, and SEO specialists in optimizing their content strategy. 
        Leveraging machine learning models and historical performance data, it analyzes key content features and predicts engagement performance 
        for various content types.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Problem Statement
    st.markdown("""
    <div class="feature-card">
        <h3>Problem Statement</h3>
        <p><strong>Challenge:</strong> Content creators and marketers struggle to optimize their content strategy due to:</p>
        <ul>
            <li>Manual tracking of content performance across multiple platforms</li>
            <li>Lack of data-driven insights into what content types perform best</li>
            <li>Difficulty identifying patterns in audience engagement</li>
            <li>Time-consuming analysis of writing style effectiveness</li>
            <li>Inability to predict content performance before publication</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features Section
    st.subheader("Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📁 File Upload</h4>
            <p>Upload your content performance dataset in CSV format, including engagement metrics and content attributes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Filter Control</h4>
            <p>Dynamically filter and explore data by content type, writing style, date range, and performance metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📈 Dynamic Graphs</h4>
            <p>Interactive visualizations of key performance metrics and content engagement patterns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Performance Insights</h4>
            <p>View content performance insights with support for interpretable visual feedback.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 Trend Analysis</h4>
            <p>Identify patterns and trends across different content types and writing styles.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>📥 Download Option</h4>
            <p>Export the analysis and insights as a CSV report for further analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Get Started Section
    st.markdown("---")
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("🚀 Ready to Begin?")
    
    st.info("""
    **To start analyzing your content performance:**
    1. Navigate to the **Begin** tab using the navigation above
    2. Upload your content performance data (CSV, Excel, or JSON format)
    3. Or use sample data to explore the dashboard features
    4. Apply filters to analyze specific segments
    5. Download insights for further analysis
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_begin_tab():
    """Display the Begin tab with data analysis"""
    
    # File Upload Section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📁 Upload Your Content Data")
    
    uploaded_file = st.file_uploader(
        "Choose your content performance file",
        type=['csv', 'xlsx', 'json'],
        help="Upload your content performance data (CSV, Excel, or JSON format)"
    )
    
    use_sample_data = st.checkbox("Use sample data for demonstration", value=True)
    
    if uploaded_file is not None:
        st.success("✅ File uploaded successfully! Proceeding to analysis...")
    elif use_sample_data:
        st.info("🔮 Using sample data to demonstrate dashboard capabilities")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize analyzer and process data
    analyzer = ContentAnalyzer()
    
    if uploaded_file is not None or use_sample_data:
        if use_sample_data:
            # Generate sample data
            analyzer.df = generate_sample_data()
            st.success("✅ Sample data loaded successfully!")
        else:
            # Load uploaded data
            analyzer.df = analyzer.load_data(uploaded_file)
            if analyzer.df is not None:
                st.success(f"✅ Successfully loaded data with {len(analyzer.df)} rows")
        
        if analyzer.df is not None:
            # Validate data and get column mapping
            column_mapping = analyzer.validate_data(analyzer.df)
            
            # Analysis Dashboard
            st.markdown("---")
            st.header("📊 Content Performance Analysis")
            
            # Sidebar filters
            st.sidebar.title("🔍 Filter Controls")
            
            # Date filter
            date_columns = ['date', 'Date', 'published_date', 'created_at']
            date_col = None
            for col in date_columns:
                if col in analyzer.df.columns:
                    date_col = col
                    break
            
            if date_col:
                min_date = analyzer.df[date_col].min().date()
                max_date = analyzer.df[date_col].max().date()
                date_range = st.sidebar.date_input(
                    "Date Range",
                    [min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
            
            # Content type filter
            content_type_columns = ['content_type', 'type', 'contentType', 'category']
            content_type_col = None
            for col in content_type_columns:
                if col in analyzer.df.columns:
                    content_type_col = col
                    break
            
            if content_type_col:
                content_types = ['All'] + list(analyzer.df[content_type_col].unique())
                selected_content_type = st.sidebar.selectbox("Content Type", content_types)
            
            # Writing style filter
            writing_style_columns = ['writing_style', 'style', 'tone']
            writing_style_col = None
            for col in writing_style_columns:
                if col in analyzer.df.columns:
                    writing_style_col = col
                    break
            
            if writing_style_col:
                writing_styles = ['All'] + list(analyzer.df[writing_style_col].unique())
                selected_writing_style = st.sidebar.selectbox("Writing Style", writing_styles)
            
            # Client filter
            client_columns = ['client', 'Client', 'customer', 'brand']
            client_col = None
            for col in client_columns:
                if col in analyzer.df.columns:
                    client_col = col
                    break
            
            if client_col:
                clients = ['All'] + list(analyzer.df[client_col].unique())
                selected_client = st.sidebar.selectbox("Client", clients)
            
            # Apply filters
            filtered_df = analyzer.df.copy()
            
            if date_col and len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                filtered_df = filtered_df[
                    (filtered_df[date_col] >= start_date) & 
                    (filtered_df[date_col] <= end_date)
                ]
            
            if content_type_col and selected_content_type != 'All':
                filtered_df = filtered_df[filtered_df[content_type_col] == selected_content_type]
            
            if writing_style_col and selected_writing_style != 'All':
                filtered_df = filtered_df[filtered_df[writing_style_col] == selected_writing_style]
            
            if client_col and selected_client != 'All':
                filtered_df = filtered_df[filtered_df[client_col] == selected_client]
            
            # Display Performance Metrics
            st.subheader("📈 Performance Overview")
            metrics = analyzer.get_performance_metrics(filtered_df, column_mapping)
            
            # Create columns for metrics
            cols = st.columns(len(metrics))
            for i, (key, value) in enumerate(metrics.items()):
                with cols[i]:
                    st.markdown(f'<div class="metric-card"><h4>{key}</h4><h3>{value}</h3></div>', unsafe_allow_html=True)
            
            # Visualizations
            st.markdown("---")
            st.subheader("📊 Performance Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Content Type Performance
                if content_type_col in filtered_df.columns:
                    views_col = column_mapping.get('views', 'views')
                    performance_by_type = filtered_df.groupby(content_type_col)[views_col].mean().reset_index()
                    
                    fig = px.bar(
                        performance_by_type,
                        x=content_type_col,
                        y=views_col,
                        title='Average Views by Content Type',
                        color=content_type_col
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Engagement by Writing Style
                writing_style_col = column_mapping.get('writing_style', 'writing_style')
                engagement_col = column_mapping.get('engagement', 'engagement_rate')
                
                if writing_style_col in filtered_df.columns and engagement_col in filtered_df.columns:
                    style_performance = filtered_df.groupby(writing_style_col)[engagement_col].mean().reset_index()
                    
                    fig = px.pie(
                        style_performance,
                        values=engagement_col,
                        names=writing_style_col,
                        title='Engagement Distribution by Writing Style'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Additional Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Time Series Analysis
                if date_col and views_col in filtered_df.columns:
                    time_series = filtered_df.groupby(date_col)[views_col].sum().reset_index()
                    
                    fig = px.line(
                        time_series,
                        x=date_col,
                        y=views_col,
                        title='Content Performance Over Time',
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Client Performance
                if client_col and engagement_col in filtered_df.columns:
                    client_performance = filtered_df.groupby(client_col)[engagement_col].mean().reset_index()
                    
                    fig = px.bar(
                        client_performance,
                        x=client_col,
                        y=engagement_col,
                        title='Average Engagement Rate by Client',
                        color=client_col
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations Section
            st.markdown("---")
            st.subheader("💡 Content Strategy Insights")
            
            engagement_col = column_mapping.get('engagement', 'engagement_rate')
            if content_type_col in filtered_df.columns and engagement_col in filtered_df.columns:
                best_type = filtered_df.groupby(content_type_col)[engagement_col].mean().idxmax()
                worst_type = filtered_df.groupby(content_type_col)[engagement_col].mean().idxmin()
                best_style = filtered_df.groupby(writing_style_col)[engagement_col].mean().idxmax() if writing_style_col in filtered_df.columns else "N/A"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="insight-card">
                        <h4>🎯 Top Performing Content Type</h4>
                        <p><strong>{}</strong> has the highest engagement rate</p>
                    </div>
                    """.format(best_type), unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="insight-card">
                        <h4>✍️ Recommended Writing Style</h4>
                        <p><strong>{}</strong> drives the most engagement</p>
                    </div>
                    """.format(best_style), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="insight-card">
                        <h4>⚠️ Content Type to Improve</h4>
                        <p><strong>{}</strong> has the lowest engagement rate</p>
                    </div>
                    """.format(worst_type), unsafe_allow_html=True)
                    
                    # Additional insight based on read time
                    if 'read_time_minutes' in filtered_df.columns and engagement_col in filtered_df.columns:
                        correlation = filtered_df['read_time_minutes'].corr(filtered_df[engagement_col])
                        insight = "Longer content performs better" if correlation > 0 else "Shorter content performs better"
                        st.markdown("""
                        <div class="insight-card">
                            <h4>📊 Content Length Insight</h4>
                            <p><strong>{}</strong></p>
                        </div>
                        """.format(insight), unsafe_allow_html=True)
            
            # Data Export
            st.markdown("---")
            st.subheader("📥 Export Results")
            
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name="content_strategy_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

def main():
    # Header Section
    st.markdown('<h1 class="main-header">Analyzing User Engagement for Content Strategy Optimization</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered tool to analyze content performance and optimize your content strategy</p>', unsafe_allow_html=True)
    
    # Navigation Tabs
    selected_nav = create_navigation()
    
    # Display content based on navigation selection
    if selected_nav == "About Tool":
        display_about_tool()
    else:  # Begin tab
        display_begin_tab()

if __name__ == "__main__":
    main()
