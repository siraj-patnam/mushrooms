import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns

def main():
    st.title("Are your mushrooms poisonous?")
    st.sidebar.title("Model Parameters")

    @st.cache_data(persist=True)
    def load_data():
        # Direct data loading instead of using ucimlrepo
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
        column_names = [
            'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
            'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring',
            'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
            'ring-type', 'spore-print-color', 'population', 'habitat'
        ]
        
        df = pd.read_csv(url, names=column_names)
        
        # Encode categorical features
        le = LabelEncoder()
        for col in df.columns:
            df[col] = le.fit_transform(df[col])
            
        return df

    df = load_data()
    
    class_names = ['edible', 'poisonous']
    
    # Display dataset overview
    if st.sidebar.checkbox("Show Dataset Information", False):
        st.subheader("Mushroom Dataset Overview")
        st.write(df.head())
        st.write("Shape:", df.shape)
        st.write("Feature names:", df.columns[:-1].tolist())
        st.write("Target distribution:")
        st.write(df['class'].value_counts())
    
    # Feature selection - improved with categorized checkboxes
    all_features = df.columns[1:].tolist()  # Exclude class column
    
    # Group features by category for easier selection
    feature_categories = {
        "Cap Features": ["cap-shape", "cap-surface", "cap-color", "bruises"],
        "Gill Features": ["gill-attachment", "gill-spacing", "gill-size", "gill-color"],
        "Stalk Features": ["stalk-shape", "stalk-root", "stalk-surface-above-ring", 
                           "stalk-surface-below-ring", "stalk-color-above-ring", 
                           "stalk-color-below-ring"],
        "Other Features": ["veil-type", "veil-color", "ring-number", "ring-type", 
                          "spore-print-color", "population", "habitat", "odor"]
    }
    
    # Create a feature selector with expandable categories
    st.sidebar.subheader("Feature Selection")
    
    # Initialize selected_features with all features
    selected_features = all_features.copy()  # Default to all features
    
    # Option to select all or clear all
    select_all = st.sidebar.checkbox("Select All Features", True)
    if not select_all:
        selected_features = []  # If "Select All" is unchecked, start with empty list
        
        # Show feature categories as expandable sections
        for category, features in feature_categories.items():
            with st.sidebar.expander(f"{category} ({len(features)})"):
                for feature in features:
                    if st.checkbox(feature, value=False, key=f"feat_{feature}"):
                        selected_features.append(feature)
    
    if len(selected_features) == 0:
        st.error("Please select at least one feature")
        return
    
    # Display the number of selected features
    st.sidebar.write(f"Selected {len(selected_features)} out of {len(all_features)} features")
    
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
            n_estimators = st.sidebar.slider("n_estimators", 10, 500, 100, 10)
            max_depth = st.sidebar.slider("max_depth", 2, 20, 5, 1)
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
        with st.spinner("Training model and generating evaluation metrics..."):
            st.subheader(f"Classifier: {classifier_name}")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Model accuracy
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Display metrics in a nicer format
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("Precision", f"{precision:.4f}")
            with col2:
                st.metric("Recall", f"{recall:.4f}")
                st.metric("F1 Score", f"{f1:.4f}")
            
            # Plot confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            st.pyplot(fig)
            
            # ROC Curve
            st.subheader("ROC Curve")
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
                st.pyplot(fig)
            else:
                st.write("ROC curve not available for this model with these settings")
            
            # Precision-Recall Curve
            st.subheader("Precision-Recall Curve")
            if hasattr(clf, "predict_proba"):
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
                pr_auc = auc(recall_curve, precision_curve)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.plot(recall_curve, precision_curve, label=f'AUC = {pr_auc:.3f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend(loc='lower left')
                st.pyplot(fig)
            else:
                st.write("Precision-Recall curve not available for this model with these settings")
            
            # Feature importance for Random Forest
            if classifier_name == "Random Forest":
                st.subheader("Feature Importance")
                feature_imp = pd.Series(clf.feature_importances_, index=selected_features).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_imp.plot.bar()
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.title('Feature Importance')
                st.pyplot(fig)

if __name__ == '__main__':
    main()
