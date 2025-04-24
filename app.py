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
import pickle
import base64

def main():
    # Create a session state to store trained models and other values
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'feature_encoders' not in st.session_state:
        st.session_state.feature_encoders = {}
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    
    # Page navigation
    page = st.sidebar.radio("Navigate", ["Train Model", "Make Prediction"])
    
    if page == "Train Model":
        train_model_page()
    else:
        prediction_page()

def train_model_page():
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
        
        # Load raw data for display
        raw_df = pd.read_csv(url, names=column_names)
        
        # Create encoded copy for model and store encoders
        df = raw_df.copy()
        encoders = {}
        for col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            
        return raw_df, df, encoders

    raw_df, df, encoders = load_data()
    # Store encoders in session state for prediction page
    st.session_state.feature_encoders = encoders
    
    # Dictionary explaining categorical values
    categorical_meanings = {
        'class': {
            'e': 'edible',
            'p': 'poisonous'
        },
        'cap-shape': {
            'b': 'bell',
            'c': 'conical',
            'x': 'convex',
            'f': 'flat',
            'k': 'knobbed',
            's': 'sunken'
        },
        'cap-surface': {
            'f': 'fibrous',
            'g': 'grooves',
            'y': 'scaly',
            's': 'smooth'
        },
        'cap-color': {
            'n': 'brown',
            'b': 'buff',
            'c': 'cinnamon',
            'g': 'gray',
            'r': 'green',
            'p': 'pink',
            'u': 'purple',
            'e': 'red',
            'w': 'white',
            'y': 'yellow'
        },
        'bruises': {
            't': 'bruises',
            'f': 'no bruises'
        },
        'odor': {
            'a': 'almond',
            'l': 'anise',
            'c': 'creosote',
            'y': 'fishy',
            'f': 'foul',
            'm': 'musty',
            'n': 'none',
            'p': 'pungent',
            's': 'spicy'
        },
        'gill-attachment': {
            'a': 'attached',
            'd': 'descending',
            'f': 'free',
            'n': 'notched'
        },
        'gill-spacing': {
            'c': 'close',
            'w': 'crowded',
            'd': 'distant'
        },
        'gill-size': {
            'b': 'broad',
            'n': 'narrow'
        },
        'gill-color': {
            'k': 'black',
            'n': 'brown',
            'b': 'buff',
            'h': 'chocolate',
            'g': 'gray',
            'r': 'green',
            'o': 'orange',
            'p': 'pink',
            'u': 'purple',
            'e': 'red',
            'w': 'white',
            'y': 'yellow'
        }
        # Note: Dictionary continues with other features but shortened for brevity
    }
    
    class_names = ['edible', 'poisonous']
    
    # Display dataset overview
    if st.sidebar.checkbox("Show Dataset Information", False):
        st.subheader("Mushroom Dataset Overview")
        
        # Show raw data before encoding
        st.write("**Original Raw Data (Before Encoding)**")
        st.write(raw_df.head())
        
        # Show encoded data
        st.write("**Encoded Data (Used for Training)**")
        st.write(df.head())
        
        st.write("Shape:", df.shape)
        st.write("Feature names:", df.columns[:-1].tolist())
        st.write("Target distribution:")
        st.write(df['class'].value_counts())
        
        # Option to show entire dataset
        if st.checkbox("Show Entire Raw Dataset"):
            st.write("**Full Raw Dataset**")
            st.dataframe(raw_df)
        
        # Option to show categorical value meanings
        if st.checkbox("Show Categorical Value Meanings"):
            st.write("**Categorical Value Meanings**")
            selected_feature = st.selectbox("Select feature to view its categorical values:", 
                                           list(categorical_meanings.keys()))
            
            if selected_feature in categorical_meanings:
                meanings_df = pd.DataFrame.from_dict(
                    categorical_meanings[selected_feature], 
                    orient='index', 
                    columns=['Meaning']
                )
                meanings_df.index.name = 'Code'
                st.write(f"**Meanings for '{selected_feature}' values:**")
                st.table(meanings_df)
            else:
                st.write("No detailed meanings available for this feature.")
    
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
    
    # Store selected features in session state for prediction page
    st.session_state.selected_features = selected_features
    
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
            
            # Store the trained model in session state
            st.session_state.model = clf
            
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
            
            # Success message and navigation tip
            st.success("Model trained successfully! Now you can go to the 'Make Prediction' page to test your model.")

def prediction_page():
    st.title("Mushroom Toxicity Prediction")
    
    # Check if model is trained
    if st.session_state.model is None:
        st.warning("Please train a model first in the 'Train Model' page.")
        return
    
    st.write("Enter mushroom characteristics to predict if it's poisonous or edible.")
    
    # Dictionary of all possible values for each feature
    feature_values = {
        'cap-shape': ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'],
        'cap-surface': ['fibrous', 'grooves', 'scaly', 'smooth'],
        'cap-color': ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'],
        'bruises': ['bruises', 'no bruises'],
        'odor': ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'],
        'gill-attachment': ['attached', 'descending', 'free', 'notched'],
        'gill-spacing': ['close', 'crowded', 'distant'],
        'gill-size': ['broad', 'narrow'],
        'gill-color': ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'],
        'stalk-shape': ['enlarging', 'tapering'],
        'stalk-root': ['bulbous', 'club', 'cup', 'equal', 'rhizomorphs', 'rooted', 'missing'],
        'stalk-surface-above-ring': ['fibrous', 'scaly', 'silky', 'smooth'],
        'stalk-surface-below-ring': ['fibrous', 'scaly', 'silky', 'smooth'],
        'stalk-color-above-ring': ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'],
        'stalk-color-below-ring': ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'],
        'veil-type': ['partial', 'universal'],
        'veil-color': ['brown', 'orange', 'white', 'yellow'],
        'ring-number': ['none', 'one', 'two'],
        'ring-type': ['cobwebby', 'evanescent', 'flaring', 'large', 'none', 'pendant', 'sheathing', 'zone'],
        'spore-print-color': ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'],
        'population': ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'],
        'habitat': ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods']
    }
    
    # Create a mapping from readable values to encoded values
    # This is needed because our model was trained on encoded values
    inverse_mapping = {}
    for feature, encoder in st.session_state.feature_encoders.items():
        if feature != 'class':  # Skip the target variable
            # Get the classes that the encoder knows
            classes = encoder.classes_
            # Create a mapping from human-readable to encoded
            if feature in feature_values:
                human_readable = feature_values[feature]
                # Find corresponding original values in encoder classes
                original_values = []
                for val in human_readable:
                    # Find the closest match in encoder classes
                    for c in classes:
                        if val.startswith(c) or c.startswith(val):
                            original_values.append(c)
                            break
                    else:
                        # If no match found, use first class
                        original_values.append(classes[0])
                
                # Create mapping from human-readable to encoded
                inverse_mapping[feature] = {hr: encoder.transform([ov])[0] 
                                          for hr, ov in zip(human_readable, original_values)}
    
    # Only show selected features from the training page
    selected_features = st.session_state.selected_features
    
    # Group features for better UX
    feature_groups = {
        "Cap Features": [f for f in selected_features if f in ["cap-shape", "cap-surface", "cap-color", "bruises"]],
        "Gill Features": [f for f in selected_features if f in ["gill-attachment", "gill-spacing", "gill-size", "gill-color"]],
        "Stalk Features": [f for f in selected_features if f.startswith("stalk")],
        "Other Features": [f for f in selected_features if not any(f.startswith(p) for p in ["cap", "gill", "stalk"])]
    }
    
# Create input form for user to select feature values
with st.form(key='prediction_form'):
    # Initialize an empty dictionary to store user inputs
    user_input = {}
    
    # Create form inputs for each feature group
    for group_name, features in feature_groups.items():
        if features:  # Only show groups that have features
            st.subheader(group_name)
            cols = st.columns(min(3, len(features)))  # Up to 3 columns
            
            for i, feature in enumerate(features):
                with cols[i % 3]:
                    # Get possible values for this feature
                    possible_values = feature_values.get(feature, ['Unknown'])
                    
                    # Create a selectbox for the feature
                    selected_value = st.selectbox(
                        f"{feature.replace('-', ' ').title()}",
                        possible_values
                    )
                    
                    # Store the selected value (encoded)
                    encoded_value = inverse_mapping.get(feature, {}).get(selected_value, 0)
                    user_input[feature] = encoded_value
    
    # Add a prominent, clearly visible submit button
    st.markdown("### Submit Prediction")
    submit_button = st.form_submit_button(
        label='PREDICT MUSHROOM TOXICITY', 
        use_container_width=True,
        type="primary"  # Makes the button more prominent
    )
    
    # Make prediction when form is submitted
    if submit_button:
        # Create a dataframe with the user input
        input_df = pd.DataFrame([user_input])
        
        # Check if we have all required features
        missing_features = set(selected_features) - set(input_df.columns)
        if missing_features:
            st.error(f"Missing features: {', '.join(missing_features)}")
            return
        
        # Ensure dataframe has the same columns in the same order as training data
        input_df = input_df[selected_features]
        
        # Make prediction
        prediction = st.session_state.model.predict(input_df)[0]
        probability = st.session_state.model.predict_proba(input_df)[0]
        
        # Display the result
        st.subheader("Prediction Result")
        result = "Edible" if prediction == 0 else "Poisonous"
        probability_value = probability[1] if prediction == 1 else probability[0]  # Probability of the predicted class
        
        # Use different colors based on the prediction
        if result == "Edible":
            st.success(f"This mushroom is predicted to be **{result}** with {probability_value:.2%} confidence.")
        else:
            st.error(f"This mushroom is predicted to be **{result}** with {probability_value:.2%} confidence.")
            st.warning("⚠️ **CAUTION**: Never eat a mushroom based solely on an app's prediction! Always consult with a mushroom expert.")
        
        # Create a nice visualization of the prediction
        fig, ax = plt.subplots(figsize=(10, 2))
        plt.barh(['Poisonous', 'Edible'], [probability[1], probability[0]], color=['#ff6b6b', '#4CAF50'])
        plt.xlim(0, 1)
        plt.xlabel('Probability')
        plt.title('Prediction Probabilities')
        for i, v in enumerate([probability[1], probability[0]]):
            plt.text(v + 0.01, i, f"{v:.2%}", va='center')
        st.pyplot(fig)
        
        # Feature importance if Random Forest
        if isinstance(st.session_state.model, RandomForestClassifier):
            st.subheader("Feature Importance for this Prediction")
            
            # Get feature importance from model
            importance = st.session_state.model.feature_importances_
            feat_importance = pd.DataFrame({
                'Feature': selected_features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            # Highlight the top features that influenced this prediction
            top_n = min(5, len(selected_features))
            top_features = feat_importance.head(top_n)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
            plt.title(f'Top {top_n} Features for This Prediction')
            st.pyplot(fig)

if __name__ == '__main__':
    main()
