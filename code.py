import streamlit as st
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

def label_encode_categorical_columns(df):
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])
    return df

def impute_missing_values(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    numerical_columns = df.select_dtypes(exclude=['object']).columns

    df_numerical = df[numerical_columns]
    numerical_imputer = SimpleImputer(strategy='mean')
    df_numerical_imputed = pd.DataFrame(numerical_imputer.fit_transform(df_numerical), columns=numerical_columns)

    df_categorical = df[categorical_columns]
    if not df_categorical.empty:
        knn_imputer_categorical = KNNImputer(n_neighbors=5)
        df_categorical_imputed = pd.DataFrame(knn_imputer_categorical.fit_transform(df_categorical), columns=categorical_columns)
    else:
        df_categorical_imputed = pd.DataFrame()

    df_imputed = pd.concat([df_categorical_imputed, df_numerical_imputed], axis=1)

    return df_imputed

def evaluate_model_condition(metric_value, metric_type):
    if metric_type == 'accuracy':
        if metric_value >= 0.8:
            return "Perfect"
        elif metric_value >= 0.7:
            return "Good"
        elif metric_value >= 0.6:
            return "Average"
        else:
            return "Bad"
    elif metric_type == 'r2':
        if metric_value >= 0.8:
            return "Perfect"
        elif metric_value >= 0.7:
            return "Good"
        elif metric_value >= 0.6:
            return "Average"
        else:
            return "Bad"
    elif metric_type == 'mse':
        if metric_value <= 0.1:
            return "Perfect"
        elif metric_value <= 1:
            return "Good"
        elif metric_value <= 10:
            return "Average"
        else:
            return "Bad"
    elif metric_type == 'mae':
        if metric_value <= 0.1:
            return "Perfect"
        elif metric_value <= 1:
            return "Good"
        elif metric_value <= 10:
            return "Average"
        else:
            return "Bad"

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if isinstance(model, (LinearRegression, DecisionTreeRegressor, RandomForestRegressor, KNeighborsRegressor, SVR)):
        st.write("### Regression Results:")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        st.write("Mean Squared Error:", mse)
        st.write("R-squared:", r2)
        st.write("Mean Absolute Error:", mae)

        st.write("Interpretation:")
        st.write("R-squared value closer to 1 indicates a good fit of the model to the data.")
        st.write("Model Condition:", evaluate_model_condition(r2, 'r2'))

    elif isinstance(model, (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, SVC, GaussianNB)):
        st.write("### Classification Results:")
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy Score:", accuracy)

        st.write("Interpretation:")
        st.write("Accuracy score closer to 1 indicates a higher percentage of correct predictions.")
        st.write("Model Condition:", evaluate_model_condition(accuracy, 'accuracy'))

        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(plt)

def perform_eda(data, x_columns, y_column):
    st.write(f"### Exploratory Data Analysis (EDA)")
    fig, axes = plt.subplots(nrows=len(x_columns), ncols=2, figsize=(12, 6 * len(x_columns)))

    for i, x_col in enumerate(x_columns):
        sns.scatterplot(x=x_col, y=y_column, data=data, ax=axes[i, 0])
        axes[i, 0].set_xlabel(x_col)
        axes[i, 0].set_ylabel(y_column)
        axes[i, 0].set_title(f"Scatter Plot: {x_col} vs {y_column}")

        sns.barplot(x=x_col, y=y_column, data=data, ax=axes[i, 1])
        axes[i, 1].set_xlabel(x_col)
        axes[i, 1].set_ylabel(y_column)
        axes[i, 1].set_title(f"Bar Graph: {x_col} vs {y_column}")

    plt.tight_layout()
    st.pyplot(fig)

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("## Original Dataset:")
    st.write(df)

    selected_columns = st.sidebar.multiselect("Select columns to drop", df.columns)

    if st.sidebar.button("Drop Selected Columns and Do Feature Engineering"):
        df_dropped = df.drop(columns=selected_columns, errors='ignore')

        df_encoded = label_encode_categorical_columns(df_dropped)

        df_imputed = impute_missing_values(df_encoded)
        st.session_state.df_imputed = df_imputed

if hasattr(st.session_state, 'df_imputed') and st.session_state.df_imputed is not None:
    
    st.write("## Encoded, Imputed, and Dropped Dataset:")
    st.write(st.session_state.df_imputed)

    # Select X and Y Data
    st.sidebar.header("Select X and Y Data")
    X_columns = st.sidebar.multiselect("Select X Columns", st.session_state.df_imputed.columns)
    y_column = st.sidebar.selectbox("Select Y Column", st.session_state.df_imputed.columns)

    # EDA Button
    if st.sidebar.button(f"Do EDA"):
        perform_eda(st.session_state.df_imputed, X_columns, y_column)

    selected_columns = X_columns + [y_column]
    combined_data = st.session_state.df_imputed[selected_columns]
    # Correlation matrix
    st.write("#### Correlation Matrix:")
    corr_matrix = combined_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    st.pyplot(plt)

    # ML Algorithm Selection
    st.sidebar.header("ML Algorithm Selection")
    selected_algorithm = st.sidebar.selectbox("Select ML Algorithm", ["Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", "KNN", "Naive Bayes", "SVM"])

    X = st.session_state.df_imputed[X_columns]
    y = st.session_state.df_imputed[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    is_continuous = y.nunique() > 10  

    model = None

    st.write(f"## Training and Evaluating {selected_algorithm}")
    if selected_algorithm == "Linear Regression" and is_continuous:
        model = LinearRegression()
    elif selected_algorithm == "Logistic Regression" and not is_continuous:
        model = LogisticRegression()
    elif selected_algorithm == "Decision Tree":
        if is_continuous:
            model = DecisionTreeRegressor()
        else:
            model = DecisionTreeClassifier()
    elif selected_algorithm == "Random Forest":
        if is_continuous:
            model = RandomForestRegressor()
        else:
            model = RandomForestClassifier()
    elif selected_algorithm == "KNN":
        if is_continuous:
            model = KNeighborsRegressor()
        else:
            model = KNeighborsClassifier()
    elif selected_algorithm == "Naive Bayes" and not is_continuous:
        model = GaussianNB()
    elif selected_algorithm == "SVM":
        if is_continuous:
            model = SVR()
        else:
            model = SVC()

    if model is not None:
        if st.sidebar.button("Train and Evaluate Model"):
            train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
    else:
        st.warning("Please choose a compatible ML algorithm for the selected target variable type.")
