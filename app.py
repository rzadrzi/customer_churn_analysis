# ============================================================
# STREAMLIT DASHBOARD FOR CUSTOMER CHURN ANALYTICS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

@st.cache_data
def load_and_prepare_data():
    """Load and clean the Telco Customer Churn dataset"""
    # Load data
    df = pd.read_csv("data.csv")

    # Clean column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Convert TotalCharges to numeric (handle blank spaces)
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    df['totalcharges'] = df['totalcharges'].fillna(0)

    # Convert Churn to binary
    df['churn'] = df['churn'].map({'No': 0, 'Yes': 1})

    # Remove customerID
    df = df.drop('customerid', axis=1)

    return df


@st.cache_resource
def train_model(df):
    """Train Random Forest model"""
    # Prepare features
    X = df.drop('churn', axis=1)
    y = df['churn']

    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=123,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return model, X_train, X_test, y_train, y_test, y_pred, y_prob


# Load data and train model
df = load_and_prepare_data()
model, X_train, X_test, y_train, y_test, y_pred, y_prob = train_model(df)

# Add predictions to dataframe
df_with_predictions = df.copy()
X_all = pd.get_dummies(df.drop('churn', axis=1), drop_first=True)
df_with_predictions['churn_probability'] = model.predict_proba(X_all)[:, 1]
df_with_predictions['predicted_churn'] = model.predict(X_all)

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("🔧 Filters")

# Contract filter
contract_options = ["All"] + sorted(df['contract'].unique().tolist())
selected_contract = st.sidebar.selectbox("Contract Type:", contract_options)

# Tenure filter
min_tenure, max_tenure = st.sidebar.slider(
    "Tenure (Months):",
    min_value=0,
    max_value=72,
    value=(0, 72)
)

# Probability threshold
probability_threshold = st.sidebar.slider(
    "Churn Probability Threshold:",
    min_value=0.5,
    max_value=0.95,
    value=0.70,
    step=0.05
)

# Apply filters
filtered_df = df_with_predictions.copy()
if selected_contract != "All":
    filtered_df = filtered_df[filtered_df['contract'] == selected_contract]

filtered_df = filtered_df[
    (filtered_df['tenure'] >= min_tenure) &
    (filtered_df['tenure'] <= max_tenure)
    ]

# High-risk customers
high_risk_df = filtered_df[filtered_df['churn_probability'] >= probability_threshold]

# ============================================================
# MAIN DASHBOARD
# ============================================================

st.title("📊 Customer Churn Analytics Dashboard")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview",
    "🤖 Model Performance",
    "⚠️ High-Risk Customers",
    "🔍 Interactive Analysis"
])

# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tab1:
    # KPI Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(filtered_df):,}",
            delta=None
        )

    with col2:
        churn_rate = filtered_df['churn'].mean() * 100
        st.metric(
            label="Churn Rate",
            value=f"{churn_rate:.1f}%",
            delta=None
        )

    with col3:
        st.metric(
            label="High-Risk Customers",
            value=f"{len(high_risk_df):,}",
            delta=None
        )

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Contract")
        contract_churn = filtered_df.groupby('contract')['churn'].mean().reset_index()
        contract_churn['churn'] = contract_churn['churn'] * 100

        fig1 = px.bar(
            contract_churn,
            x='contract',
            y='churn',
            color='contract',
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={'churn': 'Churn Rate (%)', 'contract': 'Contract Type'}
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Tenure Distribution")
        fig2 = px.histogram(
            filtered_df,
            x='tenure',
            color='churn',
            color_discrete_map={0: '#2ca02c', 1: '#d62728'},
            labels={'churn': 'Churn Status', 'tenure': 'Tenure (Months)'},
            barmode='overlay',
            opacity=0.7
        )
        fig2.update_layout(showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter plot
    st.subheader("Monthly vs Total Charges")
    fig3 = px.scatter(
        filtered_df,
        x='monthlycharges',
        y='totalcharges',
        color='churn',
        color_discrete_map={0: '#2ca02c', 1: '#d62728'},
        labels={'churn': 'Churn Status', 'monthlycharges': 'Monthly Charges', 'totalcharges': 'Total Charges'},
        opacity=0.6
    )
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================

with tab2:
    st.subheader("Confusion Matrix")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No Churn', 'Predicted: Churn'],
        y=['Actual: No Churn', 'Actual: Churn'],
        colorscale='Blues',
        showscale=False,
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 20}
    ))
    fig_cm.update_layout(height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # Model metrics
    st.subheader("Key Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")

    with col2:
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        st.metric("Precision", f"{precision:.4f}")

    with col3:
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        st.metric("Recall", f"{recall:.4f}")

    with col4:
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        st.metric("F1-Score", f"{f1:.4f}")

    st.markdown("---")

    # Variable importance
    st.subheader("Top 10 Variables Driving Churn")

    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)

    fig_imp = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig_imp.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

# ============================================================
# TAB 3: HIGH-RISK CUSTOMERS
# ============================================================

with tab3:
    st.subheader("High-Risk Customers List")
    st.write(f"Found **{len(high_risk_df)}** customers with churn probability ≥ {probability_threshold * 100:.0f}%")

    # Display table
    display_df = high_risk_df[['churn_probability', 'contract', 'tenure', 'monthlycharges', 'totalcharges']].copy()
    display_df['churn_probability'] = (display_df['churn_probability'] * 100).round(1)
    display_df.columns = ['Churn Probability (%)', 'Contract', 'Tenure (Months)', 'Monthly Charges ($)',
                          'Total Charges ($)']
    display_df = display_df.sort_values('Churn Probability (%)', ascending=False)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )

    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"high_risk_customers_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ============================================================
# TAB 4: INTERACTIVE ANALYSIS
# ============================================================

with tab4:
    st.subheader("Interactive Scatter Plot")
    st.write("Hover over points to see details. Use zoom and pan.")

    # Interactive scatter with hover
    fig_interactive = px.scatter(
        filtered_df,
        x='monthlycharges',
        y='totalcharges',
        color='churn',
        color_discrete_map={0: '#2ca02c', 1: '#d62728'},
        hover_data={
            'contract': True,
            'tenure': True,
            'churn_probability': ':.2%',
            'monthlycharges': ':.2f',
            'totalcharges': ':.2f'
        },
        labels={
            'churn': 'Churn Status',
            'monthlycharges': 'Monthly Charges ($)',
            'totalcharges': 'Total Charges ($)'
        },
        opacity=0.6
    )

    fig_interactive.update_layout(
        height=600,
        hovermode='closest'
    )

    st.plotly_chart(fig_interactive, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Built with Streamlit & Python | Data: Telco Customer Churn
    </div>
    """,
    unsafe_allow_html=True
)