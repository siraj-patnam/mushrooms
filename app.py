import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def main():
    st.title("Are your mushrooms poisonous?")
    st.sidebar.title("Model Parameters")

    @st.cache_data(persist=True)
    def load_data():
        # Direct data loading from UCI
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
        column_names = [
            'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
            'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring',
            'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
            'ring-type', 'spore-print-color', 'population', 'habitat'
        ]
        
        try:
            df = pd.read_csv(url, names=column_names)
            
            # Encode categorical features
            le = LabelEncoder()
            for col in df.columns:
                df[col] = le.fit_transform(df[col])
                
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            # Fallback - load sample data if the URL fails
            st.warning("Loading sample data instead...")
            # Create a small sample dataset
            sample_data = {
                'class': [0, 1, 0, 1, 0] * 20,
                'cap-shape': [0, 1, 2, 0, 1] * 20,
                'cap-surface': [0, 1, 0, 1, 2] * 20,
                'cap-color': [0, 1, 2, 3, 0] * 20,
                'odor': [0, 1, 0, 2, 1] * 20,
            }
            return pd.DataFrame(sample_data)

    df = load_data()
    
    class_names = ['edible', 'poisonous']
    
    # Display dataset overview
    if st.sidebar.checkbox("Show Dataset Information", False):
        st.subheader("Mushroom Dataset Overview")
        st.write(df.head())
        st.write("Shape:", df.shape)
        st.write("Target distribution:")
        st.write(df['class'].value_counts())
    
    # Feature selection
    all_features = df.columns[1:].tolist()  # Exclude class column
    selected_features = st.sidebar.multiselect(
        "Select features to include", 
        options=all_features,
        default=all_features[:3] if len(all_features) > 3 else all_features
    )
    
    if len(selected_features) == 0:
        st.error("Please select at least one feature")
        return
    
    # Split the data
    X = df[selected_features]
    y = df['class']
    
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3, 0.05)
    random_state = st.sidebar.slider("Random state", 1, 100, 42, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    st.write(f"Training set size: {X_train.shape[0]}")
    st.write(f"Test set size: {X_test.shape[0]}")
    
    # Model selection
    classifier_name = st.sidebar.selectbox(
        "Select Classifier",
        ("Support Vector Machine", "Random Forest", "Logistic Regression")
    )
    
    def add_parameter_ui(clf_name):
        params = {}
        if clf_name == "Support Vector Machine":
            C = st.sidebar.slider("C", 0.01, 10.0, 1.0, 0.1)
            kernel = st.sidebar.selectbox("Kernel", ("linear", "rbf", "poly", "sigmoid"))
            gamma = st.sidebar.selectbox("Gamma", ("scale", "auto"))
            params["C"] = C
            params["kernel"] = kernel
            params["gamma"] = gamma
            
        elif clf_name == "Random Forest":
            n_estimators = st.sidebar.slider("n_estimators", 10, 200, 100, 10)
            max_depth = st.sidebar.slider("max_depth", 2, 15, 5, 1)
            criterion = st.sidebar.selectbox("criterion", ("gini", "entropy"))
            params["n_estimators"] = n_estimators
            params["max_depth"] = max_depth
            params["criterion"] = criterion
            
        else:
            C = st.sidebar.slider("C", 0.01, 10.0, 1.0, 0.1)
            solver = st.sidebar.selectbox("solver", ("liblinear", "lbfgs", "saga"))
            max_iter = st.sidebar.slider("max_iter", 100, 1000, 100, 100)
            params["C"] = C
            params["solver"] = solver
            params["max_iter"] = max_iter
            
        return params
    
    params = add_parameter_ui(classifier_name)
    
    def get_classifier(clf_name, params):
        if clf_name == "Support Vector Machine":
            clf = SVC(C=params["C"], 
                      kernel=params["kernel"], 
                      gamma=params["gamma"],
                      probability=True)
        elif clf_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                         max_depth=params["max_depth"],
                                         criterion=params["criterion"],
                                         random_state=random_state)
        else:
            clf = LogisticRegression(C=params["C"],
                                     solver=params["solver"],
                                     max_iter=params["max_iter"],
                                     random_state=random_state)
        return clf
    
    clf = get_classifier(classifier_name, params)
    
    # Train the model
    if st.sidebar.button("Train and Evaluate"):
        with st.spinner("Training model..."):
            st.subheader(f"Classifier: {classifier_name}")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Model accuracy
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Display metrics
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("Precision", f"{precision:.4f}")
            with metrics_col2:
                st.metric("Recall", f"{recall:.4f}")
                st.metric("F1 Score", f"{f1:.4f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, 
                                index=class_names, 
                                columns=class_names)
            st.table(cm_df)
            
            # ROC and Precision-Recall data
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
                
                # ROC data
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # Create ROC curve data for plotting
                roc_data = pd.DataFrame({
                    'False Positive Rate': fpr,
                    'True Positive Rate': tpr
                })
                
                st.subheader(f"ROC Curve (AUC = {roc_auc:.3f})")
                st.line_chart(roc_data.set_index('False Positive Rate'))
                
                # Precision-Recall data
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
                pr_auc = auc(recall_curve, precision_curve)
                
                # Create Precision-Recall curve data for plotting
                pr_data = pd.DataFrame({
                    'Recall': recall_curve,
                    'Precision': precision_curve
                })
                
                st.subheader(f"Precision-Recall Curve (AUC = {pr_auc:.3f})")
                st.line_chart(pr_data.set_index('Recall'))
            
            # Feature importance for Random Forest
            if classifier_name == "Random Forest":
                st.subheader("Feature Importance")
                feature_imp = pd.DataFrame({
                    'Feature': selected_features,
                    'Importance': clf.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.bar_chart(feature_imp.set_index('Feature'))

if __name__ == '__main__':
    main()
