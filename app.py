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

# Set page configuration
st.set_page_config(
    page_title="Are your mushrooms poisonous?",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #586069;
    }
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .plot-options {
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar title with emoji
    st.sidebar.markdown("# poisonous? 🍄")
    
    # Main area title
    st.markdown('<div class="main-header">Mushroom Toxicity Classifier</div>', unsafe_allow_html=True)

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
    
    # Sidebar sections
    st.sidebar.markdown('<div class="sidebar-header">Choose Classifier</div>', unsafe_allow_html=True)
    
    # Classifier selection with clean dropdown
    classifier_name = st.sidebar.selectbox(
        "Classifier",
        ("Support Vector Machine (SVM)", "Random Forest", "Logistic Regression")
    )
    
    # Model hyperparameters section
    st.sidebar.markdown('<div class="sidebar-header">Model Hyperparameters</div>', unsafe_allow_html=True)
    
    def add_parameter_ui(clf_name):
        params = {}
        if "Support Vector Machine" in clf_name:
            # C parameter with number input and +/- buttons
            c_col1, c_col2, c_col3 = st.sidebar.columns([3, 1, 1])
            with c_col1:
                C = st.number_input("C (Regularization parameter)", 0.01, 100.0, 1.0, 0.01)
            with c_col2:
                st.write("−")
            with c_col3:
                st.write("+")
            
            # Kernel as radio buttons
            st.sidebar.write("Kernel")
            kernel = st.sidebar.radio(
                "",
                ("rbf", "linear", "poly", "sigmoid"),
                horizontal=False,
                label_visibility="collapsed"
            )
            
            # Gamma as radio buttons
            st.sidebar.write("Gamma (Kernel Coefficient)")
            gamma = st.sidebar.radio(
                "",
                ("scale", "auto"),
                horizontal=False,
                label_visibility="collapsed"
            )
            
            params["C"] = C
            params["kernel"] = kernel
            params["gamma"] = gamma
            
        elif clf_name == "Random Forest":
            n_estimators = st.sidebar.slider("Number of estimators", 10, 500, 100, 10)
            max_depth = st.sidebar.slider("Maximum depth", 2, 20, 5, 1)
            criterion = st.sidebar.radio("Criterion", ("gini", "entropy"))
            
            params["n_estimators"] = n_estimators
            params["max_depth"] = max_depth
            params["criterion"] = criterion
            
        else:
            C = st.sidebar.slider("C", 0.01, 10.0, 1.0, 0.1)
            solver = st.sidebar.radio("Solver", ("liblinear", "lbfgs", "saga"))
            max_iter = st.sidebar.slider("Maximum iterations", 100, 1000, 100, 100)
            
            params["C"] = C
            params["solver"] = solver
            params["max_iter"] = max_iter
            
        return params
    
    params = add_parameter_ui(classifier_name)
    
    # Plot selection section
    st.sidebar.markdown('<div class="sidebar-header">What metrics to plot?</div>', unsafe_allow_html=True)
    
    show_confusion_matrix = st.sidebar.checkbox("Confusion Matrix", True)
    show_roc_curve = st.sidebar.checkbox("ROC Curve")
    show_pr_curve = st.sidebar.checkbox("Precision-Recall Curve")
    
    # Options for dataset
    st.sidebar.markdown('<div class="sidebar-header">Dataset Options</div>', unsafe_allow_html=True)
    show_raw_data = st.sidebar.checkbox("Show raw data")
    
    # Feature selection - categorized 
    feature_categories = {
        "Cap Features": ["cap-shape", "cap-surface", "cap-color", "bruises"],
        "Gill Features": ["gill-attachment", "gill-spacing", "gill-size", "gill-color"],
        "Stalk Features": ["stalk-shape", "stalk-root", "stalk-surface-above-ring", 
                           "stalk-surface-below-ring", "stalk-color-above-ring", 
                           "stalk-color-below-ring"],
        "Other Features": ["veil-type", "veil-color", "ring-number", "ring-type", 
                          "spore-print-color", "population", "habitat", "odor"]
    }
    
    # Default to using all features
    all_features = df.columns[1:].tolist()  # Exclude class column
    selected_features = all_features.copy()
    
    with st.sidebar.expander("Feature Selection", expanded=False):
        # Option to select all or clear all
        select_all = st.checkbox("Select All Features", True)
        if not select_all:
            selected_features = []  # If "Select All" is unchecked, start with empty list
            
            # Show feature categories as expandable sections
            for category, features in feature_categories.items():
                with st.expander(f"{category}"):
                    for feature in features:
                        if st.checkbox(feature, value=False, key=f"feat_{feature}"):
                            selected_features.append(feature)
    
    if len(selected_features) == 0:
        st.error("Please select at least one feature")
        return
    
    # Data splitting options
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.3, 0.05)
    random_state = st.sidebar.slider("Random state", 1, 100, 42, 1)
    
    # Get classifier
    def get_classifier(clf_name, params):
        if "Support Vector Machine" in clf_name:
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
    
    # Create a prominent "Classify" button
    classify_button = st.sidebar.button("CLASSIFY", use_container_width=True)
    
    # Split the data
    X = df[selected_features]
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Display raw data if selected
    if show_raw_data:
        st.subheader("Dataset Sample")
        st.write(df.head())
        st.write(f"Total samples: {df.shape[0]}, Features: {df.shape[1]-1}")
    
    # Train and evaluate model when button is clicked
    if classify_button:
        with st.spinner("Training model and generating evaluation metrics..."):
            clf = get_classifier(classifier_name, params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Model metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Display metrics in a row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color: #3366ff;">{"%.2f" % accuracy}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Accuracy</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color: #ff6b6b;">{"%.2f" % precision}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Precision</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color: #33cc33;">{"%.2f" % recall}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">Recall</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value" style="color: #ff9900;">{"%.2f" % f1}</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-label">F1 Score</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Create visualization section
            st.markdown('<hr style="margin-top: 2rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
            
            # Confusion Matrix
            if show_confusion_matrix:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                # Create a more visually appealing confusion matrix
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Use a different colormap to match the example
                cmap = sns.color_palette(["#FFED4A", "#3B0764", "#375E97", "#4CAF50"], as_cmap=True)
                
                sns.heatmap(cm, annot=True, fmt='.1e', cmap=cmap, linewidths=1, 
                          xticklabels=class_names, yticklabels=class_names, ax=ax)
                
                plt.ylabel('True label', fontsize=12)
                plt.xlabel('Predicted label', fontsize=12)
                plt.title('Confusion Matrix', fontsize=14)
                
                # Add a colorbar
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=10)
                
                st.pyplot(fig)
            
            # ROC Curve
            if show_roc_curve and hasattr(clf, "predict_proba"):
                st.subheader("ROC Curve")
                y_prob = clf.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate', fontsize=12)
                plt.ylabel('True Positive Rate', fontsize=12)
                plt.title('Receiver Operating Characteristic', fontsize=14)
                plt.legend(loc="lower right", fontsize=10)
                plt.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
            
            # Precision-Recall Curve
            if show_pr_curve and hasattr(clf, "predict_proba"):
                st.subheader("Precision-Recall Curve")
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
                pr_auc = auc(recall_curve, precision_curve)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                plt.plot(recall_curve, precision_curve, color='green', lw=2, 
                       label=f'Precision-Recall curve (area = {pr_auc:.2f})')
                plt.xlabel('Recall', fontsize=12)
                plt.ylabel('Precision', fontsize=12)
                plt.title('Precision-Recall Curve', fontsize=14)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.legend(loc="best", fontsize=10)
                plt.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
            
            # Feature importance for Random Forest
            if classifier_name == "Random Forest":
                st.subheader("Feature Importance")
                feature_imp = pd.Series(clf.feature_importances_, index=selected_features).sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(12, len(selected_features) * 0.3 + 2))
                feature_imp.plot.barh(color='teal')
                plt.xlabel('Importance', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.title('Feature Importance', fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)

if __name__ == '__main__':
    main()
