import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

# --- Page Configuration ---
st.set_page_config(
    page_title="Fraud Detection Analysis",
    page_icon="🛡️",
    layout="wide"
)

# --- Caching Functions for Performance ---

@st.cache_data
def load_data(uploaded_file):
    """Loads data from the uploaded CSV file. Caches the result."""
    df = pd.read_csv(uploaded_file)
    return df

@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained model. Caches the resource."""
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        return model
    return None

@st.cache_data
def preprocess_labeled_data(_df):
    """Preprocesses labeled data for evaluation."""
    df_processed = _df.copy()
    known_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    df_processed = df_processed[df_processed['type'].isin(known_types)]
    if df_processed.empty:
        return df_processed
    le = LabelEncoder().fit(known_types)
    df_processed['type'] = le.transform(df_processed['type'])
    df_processed = df_processed.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    return df_processed

@st.cache_data
def preprocess_unlabeled_data(_df):
    """Preprocesses unlabeled data for prediction."""
    df_processed = _df.copy()
    known_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    df_processed = df_processed[df_processed['type'].isin(known_types)]
    if df_processed.empty:
        return df_processed
        
    le = LabelEncoder().fit(known_types)
    df_processed['type'] = le.transform(df_processed['type'])
    
    # Define the columns the model expects
    model_features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    # Drop columns that are not needed, if they exist
    cols_to_drop = ['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud']
    for col in cols_to_drop:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])
            
    # Ensure all required model features are present
    for col in model_features:
        if col not in df_processed.columns:
            st.error(f"Missing required column in the unlabeled dataset: '{col}'")
            return None
            
    return df_processed[model_features]


# --- Helper Functions for Plotting ---

def plot_transaction_counts(df, title="Transaction Counts"):
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(x='isFraud', data=df, palette="viridis", ax=ax)
    ax.set_title("Distribution of Transactions")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-Fraudulent', 'Fraudulent'])
    ax.set_ylabel("Number of Transactions")
    ax.set_xlabel("Transaction Status")
    st.pyplot(fig)

def plot_fraud_by_type(df):
    st.subheader("Fraud Rate by Transaction Type")
    fraud_by_type = df.groupby('type')['isFraud'].mean() * 100
    fraud_by_type = fraud_by_type.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=fraud_by_type.index, y=fraud_by_type.values, palette="magma", ax=ax)
    ax.set_title("Percentage of Fraudulent Transactions by Type")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_xlabel("Transaction Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

def plot_feature_importance(_model, features):
    st.subheader("Model Feature Importance")
    importances = _model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis', ax=ax)
    ax.set_title("Feature Importance from Random Forest Model")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    st.pyplot(fig)

# --- Main Application ---
def main():
    st.title("🛡️ Financial Fraud Detection")
    st.markdown("This application provides two main functionalities: analyzing a labeled dataset to evaluate model performance, and predicting fraud on new, unlabeled data.")

    model = load_model('model/fraud_detector.pkl')
    if not model:
        st.error("Model file not found! Please ensure 'fraud_detector.pkl' is in the 'model' directory.")
        return

    tab1, tab2 = st.tabs(["Analyze Labeled Dataset", "Predict on Unlabeled Dataset"])

    # --- Tab 1: Analyze Labeled Dataset ---
    with tab1:
        st.header("Evaluate Model Performance with Labeled Data")
        labeled_file = st.file_uploader("Upload a Labeled CSV file", type="csv", key="labeled")

        if labeled_file:
            df = load_data(labeled_file)
            
            if 'isFraud' not in df.columns:
                st.error("The uploaded file is missing the 'isFraud' column. Please upload a labeled dataset or use the 'Predict on Unlabeled Dataset' tab.")
                return

            df_processed = preprocess_labeled_data(df)
            
            if df_processed.empty:
                st.warning("The uploaded file contains no data with transaction types the model was trained on.")
                return

            st.subheader("Model Accuracy")
            X = df_processed.drop('isFraud', axis=1)
            y = df_processed['isFraud']
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.metric(label="Model Accuracy on Test Data", value=f"{accuracy:.4%}")

            st.header("Data Exploration and Evaluation")
            num_total = len(df)
            num_fraud = df['isFraud'].sum()
            st.metric("Total Transactions", f"{num_total:,}")
            st.metric("Fraudulent Transactions", f"{num_fraud:,}")
            
            actual_frauds = df[df['isFraud'] == 1]
            if not actual_frauds.empty:
                with st.expander("Click to view all actual fraudulent transactions from the dataset"):
                    st.dataframe(actual_frauds)
            
            plot_transaction_counts(df)
            plot_fraud_by_type(df)
            
            st.header("Model Performance Metrics (on 20% Test Sample)")
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            st.text("Classification Report:")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            with col2:
                st.subheader("ROC Curve")
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                st.write(f"**ROC AUC Score: {roc_auc:.4f}**")
                fig, ax = plt.subplots()
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            
            plot_feature_importance(model, X.columns)

    # --- Tab 2: Predict on Unlabeled Dataset ---
    with tab2:
        st.header("Find Potential Fraud in Unlabeled Data")
        unlabeled_file = st.file_uploader("Upload an Unlabeled CSV file", type="csv", key="unlabeled")

        if unlabeled_file:
            with st.spinner("Analyzing transactions..."):
                df_unlabeled = load_data(unlabeled_file)
                df_original_display = df_unlabeled.copy()
                df_unlabeled_processed = preprocess_unlabeled_data(df_unlabeled)

                if df_unlabeled_processed is not None and not df_unlabeled_processed.empty:
                    predictions = model.predict(df_unlabeled_processed)
                    
                    df_unlabeled_processed['Prediction'] = predictions
                    df_original_display = df_original_display.merge(
                        df_unlabeled_processed[['Prediction']],
                        left_index=True,
                        right_index=True,
                        how='left'
                    )
                    
                    # --- CORRECTED LOGIC TO FIND PAIRED TRANSACTIONS ---
                    predicted_frauds_df = df_original_display[df_original_display['Prediction'] == 1]
                    final_fraud_indices = set()

                    # Add all individually predicted frauds to the set first
                    for index in predicted_frauds_df.index:
                        final_fraud_indices.add(index)

                    # Now, find their pairs, even if the pair wasn't predicted as fraud
                    for index, row in predicted_frauds_df.iterrows():
                        # If a TRANSFER is flagged, find its matching CASH_OUT
                        if row['type'] == 'TRANSFER':
                            matching_cash_out = df_original_display[
                                (df_original_display['type'] == 'CASH_OUT') &
                                (df_original_display['amount'] == row['amount']) &
                                (df_original_display['step'] == row['step'])
                            ]
                            if not matching_cash_out.empty:
                                for i in matching_cash_out.index:
                                    final_fraud_indices.add(i)

                        # If a CASH_OUT is flagged, find its matching TRANSFER
                        elif row['type'] == 'CASH_OUT':
                            matching_transfer = df_original_display[
                                (df_original_display['type'] == 'TRANSFER') &
                                (df_original_display['amount'] == row['amount']) &
                                (df_original_display['step'] == row['step'])
                            ]
                            if not matching_transfer.empty:
                                for i in matching_transfer.index:
                                    final_fraud_indices.add(i)

                    fraudulent_payments = df_original_display.loc[list(final_fraud_indices)].sort_values(by=['step', 'amount'])
                    
                    st.subheader("Fraudulent Payments Found in the Dataset")
                    if not fraudulent_payments.empty:
                        st.write(f"The model identified **{len(fraudulent_payments)}** potentially fraudulent transactions (including paired transfers and cash-outs).")
                        st.dataframe(fraudulent_payments.drop(columns=['Prediction'], errors='ignore'))
                    else:
                        st.success("No fraudulent transactions were found in the provided dataset.")
                elif df_unlabeled_processed is None:
                    pass
                else:
                    st.warning("The uploaded file contains no data with transaction types the model can process.")

if __name__ == "__main__":
    main()
