from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings

warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AD ML App", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.title("🔬 Anaerobic Digestion ML App")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.sidebar.selectbox("Select Target", df.columns)

    selected_models = st.sidebar.multiselect(
        "Select Models",
        ["Random Forest", "SVR", "Linear Regression", "KNN", "XGBoost", "ANN"],
        default=["Random Forest", "XGBoost"]
    )

    train_button = st.sidebar.button("🚀 Train Models")

    # ---------------- TRAIN ----------------
    if train_button:

        if len(selected_models) == 0:
            st.warning("Please select at least one model")
            st.stop()

        # ---------------- DATA ----------------
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        feature_names = X.columns

        # ---------------- MODELS ----------------
        all_models = {
            "Random Forest": RandomForestRegressor(),
            "SVR": SVR(),
            "Linear Regression": LinearRegression(),
            "KNN": KNeighborsRegressor(),
            "XGBoost": XGBRegressor(),
            "ANN": MLPRegressor(max_iter=500)
        }

        models = {name: all_models[name] for name in selected_models}

        results = []
        trained_models = {}

        # ---------------- TRAIN LOOP ----------------
        for name, model in models.items():

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results.append([name, r2, rmse])
            trained_models[name] = model

        results_df = pd.DataFrame(results, columns=["Model", "R2 Score", "RMSE"])

        # ---------------- BEST MODEL ----------------
        best_row = results_df.loc[results_df["R2 Score"].idxmax()]
        best_model_name = best_row["Model"]
        best_model = trained_models[best_model_name]

        best_pred = best_model.predict(X_test)

        # ---------------- SAVE PREDICTIONS ----------------
        pred_df = pd.DataFrame({
            "Actual": y_test.reset_index(drop=True),
            "Best_Model_Predicted": best_pred
        })

        # ---------------- FEATURE IMPORTANCE ----------------
        feat_imp = None
        feat_imp_xgb = None

        if "Random Forest" in trained_models:
            rf_model = trained_models["Random Forest"]
            feat_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance": rf_model.feature_importances_
            }).sort_values(by="Importance", ascending=True)

        if "XGBoost" in trained_models:
            xgb_model = trained_models["XGBoost"]
            feat_imp_xgb = pd.DataFrame({
                "Feature": feature_names,
                "Importance": xgb_model.feature_importances_
            }).sort_values(by="Importance", ascending=True)

        # ---------------- TABS ----------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Model Performance",
            "📈 Predictions",
            "🔥 Feature Importance",
            "🧠 SHAP"
        ])

        # ---------------- TAB 1 ----------------
        with tab1:
            st.subheader("Model Performance")

            # Highlight best model
            def highlight_best(row):
                return ['background-color: lightgreen'] * len(row) if row["Model"] == best_model_name else [''] * len(row)

            st.dataframe(results_df.style.apply(highlight_best, axis=1))

            st.success(f"🏆 Best Model: {best_model_name}")
            st.write(f"R² Score: {best_row['R2 Score']:.4f}")
            st.write(f"RMSE: {best_row['RMSE']:.4f}")

            fig1, ax1 = plt.subplots()
            ax1.bar(results_df["Model"], results_df["R2 Score"])
            plt.xticks(rotation=30)
            st.pyplot(fig1)

        # ---------------- TAB 2 ----------------
        with tab2:

            st.subheader(f"Actual vs Predicted ({best_model_name})")

            fig2, ax2 = plt.subplots()
            ax2.scatter(y_test, best_pred)

            min_val = min(min(y_test), min(best_pred))
            max_val = max(max(y_test), max(best_pred))

            ax2.plot([min_val, max_val], [min_val, max_val], linestyle='--', color = 'red')

            ax2.set_xlabel("Actual")
            ax2.set_ylabel("Predicted")

            st.pyplot(fig2)

            st.subheader("Prediction Table")
            st.dataframe(pred_df.head())

            # Download button
            csv = pred_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

        # ---------------- TAB 3 ----------------
        with tab3:

            if feat_imp is not None:
                st.subheader("Random Forest Importance")

                fig3, ax3 = plt.subplots()
                ax3.barh(feat_imp["Feature"], feat_imp["Importance"])
                st.pyplot(fig3)

            if feat_imp_xgb is not None:
                st.subheader("XGBoost Importance")

                fig4, ax4 = plt.subplots()
                ax4.barh(feat_imp_xgb["Feature"], feat_imp_xgb["Importance"])
                st.pyplot(fig4)

        # ---------------- TAB 4 ----------------
        with tab4:
            if "Random Forest" in trained_models:
                st.subheader("SHAP Analysis (Random Forest)")

                rf_model = trained_models["Random Forest"]

                explainer = shap.TreeExplainer(rf_model)
                X_sample = X_test[:50]

                shap_values = explainer.shap_values(X_sample)

                fig5 = plt.figure()
                shap.summary_plot(
                    shap_values,
                    X_sample,
                    feature_names=feature_names,
                    show=False
                )

                st.pyplot(fig5)
            else:
                st.info("Select Random Forest to view SHAP")