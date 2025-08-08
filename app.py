import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(
    page_title="Sleep Health Dashboard",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    
    # Convert blood pressure to numeric columns
    df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    
    # Fill empty sleep disorder values with 'None'
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
    
    # Encode categorical variables
    le = LabelEncoder()
    df['Sleep Disorder Encoded'] = le.fit_transform(df['Sleep Disorder'])
    
    return df

df = load_data()

# Custom styling
st.markdown("""
<style>
    .main {
        background-color: #F5F5F5;
    }
    .sidebar .sidebar-content {
        background-color: #E1F5FE;
    }
    h1 {
        color: #0D47A1;
    }
    h2 {
        color: #1976D2;
    }
    .st-bw {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric {
        border-left: 5px solid #2196F3;
        padding-left: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar filters
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("Adjust these filters to explore the data:")

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

occupation_filter = st.sidebar.multiselect(
    "Occupation",
    options=df['Occupation'].unique(),
    default=df['Occupation'].unique()
)

bmi_filter = st.sidebar.multiselect(
    "BMI Category",
    options=df['BMI Category'].unique(),
    default=df['BMI Category'].unique()
)

# Apply filters
filtered_df = df[
    (df['Gender'].isin(gender_filter)) &
    (df['Age'] >= age_range[0]) & 
    (df['Age'] <= age_range[1]) &
    (df['Occupation'].isin(occupation_filter)) &
    (df['BMI Category'].isin(bmi_filter))
]

# Main content
st.title("😴 Sleep Health and Lifestyle Dashboard")
st.markdown("Explore how various lifestyle factors affect sleep quality and health metrics.")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Average Sleep Duration", f"{filtered_df['Sleep Duration'].mean():.1f} hours")
with col2:
    st.metric("Average Sleep Quality", f"{filtered_df['Quality of Sleep'].mean():.1f}/10")
with col3:
    st.metric("People with Sleep Disorders", f"{len(filtered_df[filtered_df['Sleep Disorder'] != 'None'])} ({len(filtered_df[filtered_df['Sleep Disorder'] != 'None'])/len(filtered_df)*100:.1f}%)")
with col4:
    st.metric("Average Daily Steps", f"{filtered_df['Daily Steps'].mean():,.0f}")

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "😴 Sleep Analysis", "❤️ Health Metrics", "👥 Demographics"])

with tab1:
    st.header("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sample Data")
        st.dataframe(filtered_df.head(10), height=300)
    
    with col2:
        st.subheader("Data Description")
        st.write(f"Total records: {len(filtered_df)}")
        st.write("""
        - **Sleep Duration**: Hours of sleep per night
        - **Quality of Sleep**: Self-rated 1-10 scale
        - **Physical Activity Level**: Minutes per day
        - **Stress Level**: Self-rated 1-10 scale
        - **BMI Category**: Weight classification
        """)
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_cols = filtered_df.select_dtypes(include=np.number).columns
    sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tab2:
    st.header("Sleep Patterns Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sleep Duration by Occupation")
        fig = px.box(filtered_df, x='Occupation', y='Sleep Duration', color='Gender',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sleep Quality vs. Stress Level")
        fig = px.scatter(filtered_df, x='Stress Level', y='Quality of Sleep', 
                         color='BMI Category', size='Sleep Duration',
                         hover_data=['Age', 'Occupation'],
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Sleep Disorder Distribution")
    # Create a filtered version that only includes rows with sleep disorders for the sunburst
    disorder_df = filtered_df[filtered_df['Sleep Disorder'] != 'None']
    if len(disorder_df) > 0:
        fig = px.sunburst(disorder_df, path=['Gender', 'BMI Category', 'Sleep Disorder'], 
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No sleep disorder data available with current filters")

with tab3:
    st.header("Health Metrics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Blood Pressure Distribution")
        fig = px.scatter(filtered_df, x='Systolic', y='Diastolic', 
                         color='Sleep Disorder', 
                         hover_data=['Age', 'Occupation', 'BMI Category'],
                         color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("BMI Category Distribution")
        fig = px.pie(filtered_df, names='BMI Category', 
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Physical Activity vs. Sleep Quality")
    fig = px.density_contour(filtered_df, x='Physical Activity Level', y='Quality of Sleep', 
                             color='Gender', marginal_x="histogram", marginal_y="histogram")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Demographic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Age Distribution by Occupation")
        fig = px.violin(filtered_df, x='Occupation', y='Age', color='Gender',
                        box=True, points="all",
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Daily Steps by BMI Category")
        fig = px.box(filtered_df, x='BMI Category', y='Daily Steps', color='Gender',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Occupation Distribution")
    # Fix for the bar chart - properly name the columns after value_counts()
    occupation_counts = filtered_df['Occupation'].value_counts().reset_index()
    occupation_counts.columns = ['Occupation', 'Count']  # Rename columns explicitly
    fig = px.bar(occupation_counts, 
                 x='Occupation', y='Count',
                 color='Occupation',
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
### Insights:
- People with higher physical activity levels tend to have better sleep quality
- Stress levels are negatively correlated with sleep quality
- Certain occupations (like nurses) show higher prevalence of sleep disorders
- BMI category has a significant impact on sleep metrics
""")