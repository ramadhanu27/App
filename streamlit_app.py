import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold  # Tambahkan KFold di sini
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set judul aplikasi
st.title("Aplikasi Streamlit dengan Visualisasi & Machine Learning")

# Fungsi untuk memproses file yang diupload
def load_data(file):
    data = pd.read_csv(file)
    return data

# Bagian upload file
st.header("Upload File CSV")
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file is not None:
    # Menampilkan isi file yang diupload
    data = load_data(uploaded_file)
    st.write("Dataframe yang diupload:")
    st.dataframe(data)

    # Statistik dasar
    st.write("Statistik Deskriptif:")
    st.write(data.describe())

    # Pilih jenis grafik yang diinginkan
    st.header("Visualisasi Data")
    plot_type = st.selectbox(
        "Pilih jenis grafik yang ingin ditampilkan:",
        ["Scatter Plot", "Histogram", "Box Plot", "Correlation Heatmap", "Line Plot"]
    )

    columns = data.columns.tolist()

    if plot_type == "Scatter Plot":
        # Scatter plot
        x_axis = st.selectbox("Pilih kolom untuk sumbu X", columns)
        y_axis = st.selectbox("Pilih kolom untuk sumbu Y", columns)

        if x_axis and y_axis:
            fig, ax = plt.subplots()
            ax.scatter(data[x_axis], data[y_axis], color="green")
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"Scatter Plot {x_axis} vs {y_axis}")
            st.pyplot(fig)

    elif plot_type == "Histogram":
        # Histogram
        selected_column = st.selectbox("Pilih kolom untuk histogram", columns)
        
        if selected_column:
            fig, ax = plt.subplots()
            sns.histplot(data[selected_column], kde=True, ax=ax, color="blue")
            ax.set_title(f"Histogram of {selected_column}")
            st.pyplot(fig)

    elif plot_type == "Box Plot":
        # Box plot
        selected_column = st.selectbox("Pilih kolom untuk box plot", columns)
        
        if selected_column:
            fig, ax = plt.subplots()
            sns.boxplot(data=data[selected_column], ax=ax)
            ax.set_title(f"Box Plot of {selected_column}")
            st.pyplot(fig)
    
    elif plot_type == "Correlation Heatmap":
        # Correlation Heatmap
        corr = data.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    elif plot_type == "Line Plot":
        # Line plot
        x_axis = st.selectbox("Pilih kolom untuk sumbu X (Line Plot)", columns)
        y_axis = st.selectbox("Pilih kolom untuk sumbu Y (Line Plot)", columns)

        if x_axis and y_axis:
            fig, ax = plt.subplots()
            ax.plot(data[x_axis], data[y_axis], marker="o", color="purple")
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"Line Plot {x_axis} vs {y_axis}")
            st.pyplot(fig)
    
    # Tambahkan bagian Machine Learning
    st.header("Machine Learning Model")
    ml_task = st.selectbox(
        "Pilih tipe Machine Learning:",
        ["Regresi", "Klasifikasi"]
    )

    if ml_task == "Regresi":
        ml_algorithm = st.selectbox(
            "Pilih algoritma Regresi:",
            ["Linear Regression", "Decision Tree", "Random Forest"]
        )

        target_column = st.selectbox("Pilih kolom target (Y)", columns)
        feature_columns = st.multiselect("Pilih kolom fitur (X)", columns)

        # Input hyperparameter untuk regresi
        if ml_algorithm == "Decision Tree":
            max_depth_dt = st.number_input("Max Depth (Decision Tree)", min_value=1, max_value=100, value=10)
        elif ml_algorithm == "Random Forest":
            n_estimators_rf = st.number_input("Jumlah Estimator (Random Forest)", min_value=10, max_value=500, value=100)
            max_depth_rf = st.number_input("Max Depth (Random Forest)", min_value=1, max_value=100, value=10)

        # Input untuk K-Fold
        k_fold = st.number_input("Jumlah Fold untuk K-Fold Cross Validation", min_value=2, max_value=20, value=5)

        if target_column and feature_columns:
            X = data[feature_columns]
            y = data[target_column]

            # Pilih algoritma regresi
            if ml_algorithm == "Linear Regression":
                model = LinearRegression()
            elif ml_algorithm == "Decision Tree":
                model = DecisionTreeRegressor(max_depth=max_depth_dt)
            elif ml_algorithm == "Random Forest":
                model = RandomForestRegressor(n_estimators=n_estimators_rf, max_depth=max_depth_rf)

            # K-Fold Cross Validation
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
            cv_scores_mse = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
            cv_scores_r2 = cross_val_score(model, X, y, scoring='r2', cv=kf)

            # Menampilkan hasil evaluasi
            st.write(f"**Model: {ml_algorithm}**")
            st.write(f"Rata-rata Mean Squared Error (MSE) dari {k_fold}-Fold CV: {-cv_scores_mse.mean()}")
            st.write(f"Rata-rata R2 Score dari {k_fold}-Fold CV: {cv_scores_r2.mean()}")

            # Split data untuk visualisasi hasil prediksi
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Visualisasi hasil prediksi vs nilai asli
            st.write("Plot Prediksi vs Nilai Asli")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, color="blue")
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel("Nilai Asli")
            ax.set_ylabel("Prediksi")
            ax.set_title(f"{ml_algorithm} - Prediksi vs Nilai Asli")
            st.pyplot(fig)

    elif ml_task == "Klasifikasi":
        ml_algorithm = st.selectbox(
            "Pilih algoritma Klasifikasi:",
            ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)"]
        )

        target_column = st.selectbox("Pilih kolom target (Y)", columns)
        feature_columns = st.multiselect("Pilih kolom fitur (X)", columns)

        # Input hyperparameter untuk klasifikasi
        if ml_algorithm == "K-Nearest Neighbors (KNN)":
            n_neighbors_knn = st.number_input("Jumlah Tetangga (KNN)", min_value=1, max_value=50, value=5)
        elif ml_algorithm == "Support Vector Machine (SVM)":
            kernel_svm = st.selectbox("Kernel (SVM)", ["linear", "poly", "rbf", "sigmoid"])

        # Input untuk K-Fold
        k_fold = st.number_input("Jumlah Fold untuk K-Fold Cross Validation", min_value=2, max_value=20, value=5)

        if target_column and feature_columns:
            X = data[feature_columns]
            y = data[target_column]

            # Pilih algoritma klasifikasi
            if ml_algorithm == "Logistic Regression":
                model = LogisticRegression()
            elif ml_algorithm == "K-Nearest Neighbors (KNN)":
                model = KNeighborsClassifier(n_neighbors=n_neighbors_knn)
            elif ml_algorithm == "Support Vector Machine (SVM)":
                model = SVC(kernel=kernel_svm)

            # K-Fold Cross Validation
            kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
            cv_scores_acc = cross_val_score(model, X, y, scoring='accuracy', cv=kf)

            # Menampilkan hasil evaluasi
            st.write(f"**Model: {ml_algorithm}**")
            st.write(f"Rata-rata Akurasi dari {k_fold}-Fold CV: {cv_scores_acc.mean()}")

            # Split data untuk visualisasi hasil prediksi
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Menampilkan metrik klasifikasi
            st.write(f"Akurasi: {accuracy_score(y_test, y_pred)}")
            st.write(f"Precision: {precision_score(y_test, y_pred, average='macro')}")
            st.write(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
            st.write(f"F1 Score: {f1_score(y_test, y_pred, average='macro')}")

            # Visualisasi matriks kebingungan
            st.write("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{ml_algorithm} - Confusion Matrix")
            st.pyplot(fig)

else:
    st.write("Silakan upload file CSV untuk melihat analisis data.")

# Footer aplikasi
st.write("Aplikasi analisis data selesai!")
