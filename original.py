import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pydp.algorithms.laplacian import BoundedMean
from sklearn.neighbors import NearestNeighbors

from scipy.stats import wasserstein_distance

try:
    from sdv.tabular import CTGAN  # For newer versions
except ImportError:
    try:
        from sdv.single_table import CTGAN  # For some intermediate versions
    except ImportError:
        from sdv.tabular.models import CTGAN  # For older versions

from sdv.evaluation import evaluate
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# Define handle_nans function first
def handle_nans(df):
    """Handle NaN values in a dataframe with detailed reporting"""
    # Check for NaNs
    nan_count = df.isna().sum().sum()
    if nan_count == 0:
        return df, False  # No NaNs found

    # Display NaN statistics by column
    st.warning(f"⚠️ Dataset contains {nan_count} missing values")

    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        st.write("Missing values by column:")
        nan_stats = pd.DataFrame(
            {
                "Column": nan_cols,
                "Missing Values": [df[col].isna().sum() for col in nan_cols],
                "Percentage": [
                    round(df[col].isna().mean() * 100, 2) for col in nan_cols
                ],
            }
        )
        st.dataframe(nan_stats)

    # NaN handling options
    nan_strategy = st.radio(
        "How would you like to handle missing values?",
        [
            "Drop rows with any missing values",
            "Fill numerical with mean, categorical with mode",
            "Keep missing values (not recommended)",
        ],
    )

    # Create a copy to avoid modifying the original
    df_clean = df.copy()

    if nan_strategy == "Drop rows with any missing values":
        original_shape = df_clean.shape
        df_clean = df_clean.dropna()
        st.write(
            f"Dropped {original_shape[0] - df_clean.shape[0]} rows with missing values"
        )

    elif nan_strategy == "Fill numerical with mean, categorical with mode":
        # Detect features first
        categorical_features = df_clean.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        numerical_features = df_clean.select_dtypes(include=["number"]).columns.tolist()

        for col in df_clean.columns:
            if col in numerical_features:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            else:
                df_clean[col] = df_clean[col].fillna(
                    df_clean[col].mode()[0]
                    if not df_clean[col].mode().empty
                    else "Unknown"
                )
        st.write("✅ Filled missing values with mean/mode")

    return df_clean, True  # Return cleaned df and flag that NaNs were found


# 📌 UI Setup
st.title("🔐 Privacy Attack App for Synthetic Data")
st.sidebar.header("Configuration")

# Fix the duplicate slider issue
# 1. Keep the slider at the top of the script
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
epsilon = st.sidebar.slider(
    "Epsilon (Privacy Budget)", 0.1, 10.0, 1.0, key="epsilon_slider"
)

# Add training configuration section
st.sidebar.subheader("Training Configuration")
ctgan_epochs = st.sidebar.slider(
    "CTGAN Max Epochs", 10, 5000, 50, step=10, key="ctgan_epochs_slider"
)
gan_epochs = st.sidebar.slider(
    "GAN Epochs", 10, 200, 5000, step=10, key="gan_epochs_slider"
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Apply NaN handling
    df, had_nans = handle_nans(df)

    st.write("📂 Uploaded Dataset (after handling missing values):")
    st.write(df.head())

    # Auto-detect categorical & numerical features
    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_features = df.select_dtypes(include=["number"]).columns.tolist()

    st.sidebar.write(f"**Categorical Features:** {categorical_features}")
    st.sidebar.write(f"**Numerical Features:** {numerical_features}")

    # Add a button to generate synthetic data
    generate_button = st.button("🚀 Generate Synthetic Data")

    if generate_button:
        # 2️⃣ Generate Synthetic Data
        st.subheader("🛠️ Generate Synthetic Data")
        if len(categorical_features) > 0:
            # Use CTGAN for mixed datasets
            st.write("✅ Using **CTGAN** for synthetic data generation.")
            ctgan = CTGAN(
                epochs=10, verbose=True
            )  # Set a small number of epochs initially

            # Train with early stopping
            best_score = 0
            best_model = None
            patience_counter = 0
            max_patience = 5
            max_epochs = ctgan_epochs  # Use the value from the slider

            # First fit to initialize the model
            ctgan.fit(df)

            with st.spinner("Training model with early stopping..."):
                progress_bar = st.progress(0)
                for i in range(max_epochs // 10):
                    # Train for a few more epochs
                    ctgan.fit(df)

                    # Update progress bar
                    progress = (i + 1) / (max_epochs // 10)
                    progress_bar.progress(progress)

                    # Sample and evaluate
                    try:
                        sample_data = ctgan.sample(min(1000, len(df)))

                        # Basic evaluation - compare distributions
                        score = 0
                        for col in numerical_features:
                            if col in sample_data.columns:
                                # Calculate similarity using Wasserstein distance
                                dist = wasserstein_distance(
                                    df[col].values, sample_data[col].values
                                )
                                # Convert to similarity score (lower distance = higher score)
                                sim = 1 / (1 + dist)
                                score += sim

                        # Normalize score
                        score = (
                            score / len(numerical_features) if numerical_features else 0
                        )

                        # Print to terminal instead of UI
                        print(f"Iteration {i+1}, Score: {score:.4f}")

                        if score > best_score:
                            best_score = score
                            best_model = copy.deepcopy(ctgan)
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        if patience_counter >= max_patience:
                            print(f"Early stopping at iteration {i+1}")
                            break

                    except Exception as e:
                        print(f"Error during evaluation: {str(e)}")
                        continue

            # Use the best model if found, otherwise use the current model
            if best_model is not None:
                ctgan = best_model
                st.success(f"✅ Found best model with score: {best_score:.4f}")

            synthetic_data = ctgan.sample(len(df))
        else:
            # Use Gaussian Sampling for purely numerical data
            st.write("✅ Using **Gaussian Sampling** for synthetic data generation.")
            synthetic_data = df[numerical_features].apply(
                lambda x: np.random.normal(x.mean(), x.std(), len(df))
            )

        st.write("📊 Synthetic Data Preview:")
        st.write(synthetic_data.head())

        # Add download button for synthetic data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode("utf-8")

        csv = convert_df_to_csv(synthetic_data)
        st.download_button(
            label="Download Synthetic Data as CSV",
            data=csv,
            file_name="synthetic_data.csv",
            mime="text/csv",
            key="download-csv",
        )

        # Add a horizontal line for visual separation
        st.markdown("---")

        # 3️⃣ Membership Inference Attack
        st.subheader("🚀 Membership Inference Attack")

        # Make sure we're using the same features for both datasets
        common_numerical_features = [
            col for col in numerical_features if col in synthetic_data.columns
        ]

        if len(common_numerical_features) > 0:
            # Create labels
            y_orig = np.ones(df.shape[0])
            y_synth = np.zeros(synthetic_data.shape[0])

            # Extract features using only common numerical columns
            try:
                # Convert to same data type to avoid dimension issues
                X_orig = df[common_numerical_features].astype(float).values
                X_synth = synthetic_data[common_numerical_features].astype(float).values

                # Check shapes before stacking
                st.write(f"Original data shape: {X_orig.shape}")
                st.write(f"Synthetic data shape: {X_synth.shape}")

                # Stack the arrays
                X = np.vstack((X_orig, X_synth))
                y = np.hstack((y_orig, y_synth))

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                attack_acc = accuracy_score(y_test, y_pred)

                st.write(f"**Attack Accuracy: {attack_acc:.4f}**")
                st.text(classification_report(y_test, y_pred))
            except Exception as e:
                st.error(f"Error during membership inference attack: {str(e)}")
                st.info(
                    "This error often occurs when data types don't match between original and synthetic data"
                )
        else:
            st.error(
                "No common numerical features found between original and synthetic data"
            )

        # 4️⃣ GAN-Based Inversion Attack
        st.subheader("🤖 GAN-Based Inversion Attack")

        # Make sure we're only using numerical features for the GAN
        if len(common_numerical_features) > 0:

            class SimpleGAN(nn.Module):
                def __init__(self, input_dim):
                    super(SimpleGAN, self).__init__()
                    self.generator = nn.Sequential(
                        nn.Linear(input_dim, 10),
                        nn.ReLU(),
                        nn.Linear(10, input_dim),
                    )
                    self.discriminator = nn.Sequential(
                        nn.Linear(input_dim, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1),
                        nn.Sigmoid(),
                    )

                def generate(self, noise):
                    return self.generator(noise)

            try:
                # Train a GAN to "reverse-engineer" synthetic data
                device = "cuda" if torch.cuda.is_available() else "cpu"

                # Use only numerical features for GAN
                input_dim = len(common_numerical_features)
                gan = SimpleGAN(input_dim).to(device)
                optimizer = optim.Adam(gan.generator.parameters(), lr=0.01)
                criterion = nn.MSELoss()

                # Convert data to tensors, using only numerical features
                synthetic_tensor = torch.tensor(
                    synthetic_data[common_numerical_features].values,
                    dtype=torch.float32,
                ).to(device)

                # Train GAN
                num_epochs = gan_epochs  # Use the value from the slider
                with st.spinner("Training GAN..."):
                    progress_bar = st.progress(0)
                    for epoch in range(num_epochs):
                        optimizer.zero_grad()
                        noise = torch.randn((len(df), input_dim)).to(device)
                        fake_data = gan.generate(noise)
                        loss = criterion(fake_data, synthetic_tensor)
                        loss.backward()
                        optimizer.step()

                        # Update progress bar
                        progress = (epoch + 1) / num_epochs
                        progress_bar.progress(progress)

                        # Print to terminal instead of UI
                        if (epoch + 1) % 10 == 0:
                            print(
                                f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}"
                            )

                # Evaluate GAN similarity
                with torch.no_grad():
                    noise = torch.randn((len(df), input_dim)).to(device)
                    synth_reconstructed = gan.generate(noise).cpu().detach().numpy()

                gan_distance = wasserstein_distance(
                    synthetic_data[common_numerical_features].values.flatten(),
                    synth_reconstructed.flatten(),
                )

                st.write(
                    f"**GAN Reconstruction Similarity (Wasserstein Distance): {gan_distance:.4f}**"
                )

                if gan_distance < 1.0:
                    st.warning(
                        "⚠️ LOW DISTANCE: GAN can closely reconstruct data (privacy risk)"
                    )
                elif gan_distance < 5.0:
                    st.info("ℹ️ MODERATE DISTANCE: Some reconstruction possible")
                else:
                    st.success(
                        "✅ HIGH DISTANCE: Difficult to reconstruct data (good privacy)"
                    )

            except Exception as e:
                st.error(f"Error during GAN attack: {str(e)}")
                st.info(
                    "This error might occur due to incompatible data types or dimensions"
                )
        else:
            st.error("Cannot run GAN attack: No common numerical features available")

        # 5️⃣ Differential Privacy Evaluation
        st.subheader("🛡️ Differential Privacy")

        # Make sure we're only using common numerical features
        if len(common_numerical_features) > 0:
            try:
                # Better differential privacy with epsilon calibration
                def apply_dp_with_calibration(data, epsilon, sensitivity=None):
                    """Apply differential privacy with automatic sensitivity calibration"""
                    if sensitivity is None:
                        # Estimate sensitivity as the range of each column
                        sensitivity = data.max() - data.min()

                    # Scale epsilon by column sensitivity
                    scaled_epsilon = epsilon / sensitivity

                    # Apply Laplace mechanism
                    dp_data = data.copy()
                    for col in dp_data.columns:
                        noise_scale = 1.0 / scaled_epsilon[col]
                        noise = np.random.laplace(0, noise_scale, size=len(dp_data))
                        dp_data[col] += noise

                    return dp_data

                # Apply calibrated DP
                dp_synthetic = apply_dp_with_calibration(
                    synthetic_data[common_numerical_features], epsilon
                )

                # Calculate Wasserstein distance
                dp_distance = wasserstein_distance(
                    df[common_numerical_features].values.flatten(),
                    dp_synthetic.values.flatten(),
                )

                st.write(
                    f"**Differential Privacy Wasserstein Distance:** {dp_distance:.4f}"
                )

                # Interpret the results
                if dp_distance < 1.0:
                    st.success("✅ LOW IMPACT: DP minimally affects utility")
                elif dp_distance < 5.0:
                    st.info(
                        "ℹ️ MODERATE IMPACT: DP provides good privacy-utility balance"
                    )
                else:
                    st.warning("⚠️ HIGH IMPACT: DP significantly alters distributions")

                # Visualization of DP impact
                st.subheader("Differential Privacy Impact Visualization")

                # Select a feature to visualize
                feature_to_viz = st.selectbox(
                    "Select feature to visualize:", common_numerical_features
                )

                # Create plot
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.kdeplot(df[feature_to_viz], label="Original", ax=ax)
                sns.kdeplot(synthetic_data[feature_to_viz], label="Synthetic", ax=ax)
                sns.kdeplot(dp_synthetic[feature_to_viz], label="DP Synthetic", ax=ax)
                ax.set_title(f"Impact of Differential Privacy on {feature_to_viz}")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error applying differential privacy: {str(e)}")
                st.info(
                    "This error might be due to incompatible data types or dimensions"
                )
        else:
            st.error(
                "Cannot apply differential privacy: No common numerical features available"
            )

        # 6️⃣ Visualization
        st.subheader("📈 Visualization")

        df_plot = pd.DataFrame(X, columns=numerical_features)
        df_plot["Type"] = np.where(y == 1, "Original", "Synthetic")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(data=df_plot, x=numerical_features[0], hue="Type", fill=True, ax=ax)
        st.pyplot(fig)

        # 🎯 Final Summary
        st.subheader("🏆 Comparison of Privacy Attacks")
        st.write(f"🔹 **Membership Attack Accuracy:** {attack_acc:.4f}")
        st.write(f"🔹 **GAN Reconstruction Similarity:** {gan_distance:.4f}")
        st.write(f"🔹 **Differential Privacy Impact:** {dp_distance:.4f}")

        if attack_acc > 0.75:
            st.warning(
                "⚠️ Membership inference attack is successful! Synthetic data might be leaking information."
            )

        if gan_distance < 5.0:
            st.warning(
                "⚠️ GAN-based inversion was able to generate similar data! Potential privacy risk."
            )

        if dp_distance > 10.0:
            st.success("✅ Differential privacy is making data more robust.")

        # K-anonymity approximation
        def estimate_k_anonymity(original, synthetic, k=5):
            # Find k nearest neighbors for each synthetic record
            nbrs = NearestNeighbors(n_neighbors=k).fit(original)
            distances, _ = nbrs.kneighbors(synthetic)

            # Average distance to k-th neighbor
            avg_distance = np.mean(distances[:, -1])
            return avg_distance

        k_anonymity_score = estimate_k_anonymity(
            df[common_numerical_features].values,
            synthetic_data[common_numerical_features].values,
        )

        st.write(f"**Estimated k-anonymity distance:** {k_anonymity_score:.4f}")
        if k_anonymity_score < 0.1:
            st.warning("⚠️ Low k-anonymity distance indicates potential privacy risk")
        else:
            st.success(
                "✅ Good k-anonymity distance indicates better privacy protection"
            )

        # Add utility metrics to evaluate synthetic data quality
        def evaluate_ml_utility(original_data, synthetic_data, target_col):
            # Split original data
            X_orig = original_data.drop(columns=[target_col])
            y_orig = original_data[target_col]

            # Split synthetic data
            X_synth = synthetic_data.drop(columns=[target_col])
            y_synth = synthetic_data[target_col]

            # Train on original, test on synthetic
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_orig, y_orig)
            y_pred = model.predict(X_synth)

            # Calculate R²
            r2 = r2_score(y_synth, y_pred)
            return r2

        # Select a target column
        if len(common_numerical_features) > 1:
            target_col = common_numerical_features[0]
            utility_r2 = evaluate_ml_utility(
                df[common_numerical_features],
                synthetic_data[common_numerical_features],
                target_col,
            )
            st.write(f"**ML Utility Score (R²):** {utility_r2:.4f}")

st.sidebar.text("Developed by CDAO Intelligence Team")
