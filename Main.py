from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
import os
from PIL import Image, ImageTk
from joblib import load
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc

import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, label_binarize, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.initializers import GlorotUniform

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")



# Setting up directories
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


accuracy = []
precision = []
recall = []
fscore = []


def load_dataset():
    """Load the dataset from a CSV file selected via file dialog."""
    filepath = filedialog.askopenfilename(
        initialdir=".",
        title="Select CSV File",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    if filepath:
        return pd.read_csv(filepath)
    else:
        print("No file selected.")
        return None

def preprocess_data(df, is_train=True, label_encoders=None):

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Drop unnamed cols
    df.drop(columns=["Timestamp"], inplace=True)

    if is_train:
        label_encoders = {}

        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        if label_encoders is None:
            raise ValueError("label_encoders must be provided for test/inference.")

        for col in df.select_dtypes(include='object').columns:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = le.transform(df[col].astype(str))
            else:
                raise ValueError(f"Missing encoder for column: {col}")

    df = df.fillna(df.mean(numeric_only=True))

    if is_train:
        X = df.drop(columns=['Dropped Connection'])
        y = df['Dropped Connection']
        return X, y, label_encoders
    else:
        return df


def perform_eda(X, y):
    os.makedirs("results", exist_ok=True)
    sns.set(style="whitegrid")
    
    #plt.figure(figsize=(18, 12))

    # 1. Count Plot of Target (Dropped Connection)
    #plt.subplot(2, 3, 1)
    sns.countplot(x=y, palette='viridis')
    plt.title("Distribution of Dropped Connections")
    plt.xlabel("Dropped Connection (False=0, True=1)")
    plt.ylabel("Count")
    plt.show()
    plt.savefig("results/Distribution of Dropped Connections.png")
    
    # 2. Distribution of Signal Strength grouped by Target
    #plt.subplot(2, 3, 2)
    sns.kdeplot(x = X["Signal Strength (dBm)"], hue = y, fill=True, common_norm=False, palette="Set2")
    plt.title("Signal Strength vs Dropped Connection")
    plt.xlabel("Signal Strength (dBm)")
    plt.ylabel("Density")
    plt.savefig("results/Signal Strength vs Dropped Connection.png")

    plt.show()
    # 3. Boxplot of Download Speed vs Dropped Connection
   # plt.subplot(2, 3, 3)
    sns.boxplot(x=y, y=X["Download Speed (Mbps)"], palette="coolwarm")
    plt.title("Download Speed vs Dropped Connection")
    plt.xlabel("Dropped Connection")
    plt.ylabel("Download Speed (Mbps)")
    plt.show()
    plt.savefig("results/Download Speed vs Dropped Connection.png")

    
    # 4. Heatmap of Numeric Feature Correlations with Target
    #plt.subplot(2, 3, 4)
    df_corr = X.copy()
    df_corr["Dropped Connection"] = y.astype(int)
    corr = df_corr.select_dtypes(include=["number"]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap with Dropped Connection")
    plt.show()
    plt.savefig("results/Correlation heatmap with Dropped Connections")

    # 5. Stacked Bar Plot: Network Type vs Dropped Connection
    #plt.subplot(2, 3, 5)
    net_counts = pd.crosstab(X["Network Type"], y, normalize="index")
    net_counts.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="Paired")
    plt.title("Proportion of Dropped Connections by Network Type")
    plt.xlabel("Network Type")
    plt.ylabel("Proportion")
    plt.show()
    plt.savefig("results/Proportion of Dropped Connections by Network Type")
    

    # 6. Pie Chart of Carrier Distribution
    #plt.subplot(2, 3, 6)
    carrier_counts = X["Carrier"].value_counts()
    plt.pie(carrier_counts, labels=carrier_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title("Carrier Distribution")

    #plt.tight_layout()
    plt.savefig("results/Carrier Distribution.png")
    plt.show()

# Dataframes to store results
metrics_df = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
class_report_df = pd.DataFrame()
class_performance_dfs = {}  # Dictionary to store dataframes for each class

if not os.path.exists('results'):
    os.makedirs('results')
    
    
def Calculate_Metrics(algorithm, predict, y_test, y_score):
    global metrics_df, class_report_df, class_performance_dfs
    
    categories = labels
    
    # Calculate overall metrics
    a = accuracy_score(y_test, predict) * 100
    p = precision_score(y_test, predict, average='macro') * 100
    r = recall_score(y_test, predict, average='macro') * 100
    f = f1_score(y_test, predict, average='macro') * 100

    # Append to global lists
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    # Create metrics dataframe entry
    metrics_entry = pd.DataFrame({
        'Algorithm': [algorithm],
        'Accuracy': [a],
        'Precision': [p],
        'Recall': [r],
        'F1-Score': [f]
    })
    metrics_df = pd.concat([metrics_df, metrics_entry], ignore_index=True)
    
    # Console output
       
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, predict)
    
    # Classification report
    CR = classification_report(y_test, predict, target_names=[str(c) for c in categories], output_dict=True)    
    CR1 = classification_report(y_test, predict, target_names=[str(c) for c in categories])
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR1) +"\n\n")

    
    # Classification report dataframe
    cr_df = pd.DataFrame(CR).transpose()
    cr_df['Algorithm'] = algorithm
    class_report_df = pd.concat([class_report_df, cr_df], ignore_index=False)
    
    # Class-specific performance dataframes
    for category in categories:
        class_entry = pd.DataFrame({
            'Algorithm': [algorithm],
            'Precision': [CR[str(category)]['precision'] * 100],
            'Recall': [CR[str(category)]['recall'] * 100],
            'F1-Score': [CR[str(category)]['f1-score'] * 100],
            'Support': [CR[str(category)]['support']]
        })
        
        if str(category) not in class_performance_dfs:
            class_performance_dfs[str(category)] = pd.DataFrame(columns=['Algorithm', 'Precision', 'Recall', 'F1-Score', 'Support'])
        
        class_performance_dfs[str(category)] = pd.concat([class_performance_dfs[str(category)], class_entry], ignore_index=True)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(conf_matrix, xticklabels=categories, yticklabels=categories, annot=True, cmap="viridis", fmt="g")
    ax.set_ylim([0, len(categories)])
    plt.title(f"{algorithm} Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png")
    plt.show()

    if y_score is not None:

            # take probability of positive class (binary)
            if y_score.ndim > 1:
                y_prob = y_score[:, 1]
            else:
                y_prob = y_score

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8,8))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
            plt.plot([0,1], [0,1], linestyle='--')

            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{algorithm} ROC Curve")
            plt.legend(loc="lower right")

            plt.savefig(f"results/{algorithm.replace(' ', '_')}_roc_curve.png")
            plt.show()

            text.insert(END, algorithm + " ROC AUC : " + str(roc_auc) + "\n\n")
        

def train_knn_classifier(X_train, y_train, X_test, y_test, n_neighbors=2):
    model_path = os.path.join(MODEL_DIR, f'knn_classifier_{n_neighbors}.joblib')

    if os.path.exists(model_path):
        model = load(model_path)
    else:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train_smote, y_train_smote)
        dump(model, model_path)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    Calculate_Metrics(f"KNN Classifier (k={n_neighbors})", y_pred, y_test, y_score)


def train_catboost_classifier(X_train, y_train, X_test, y_test):
    """Train and evaluate CatBoost Classifier model."""
    model_path = os.path.join(MODEL_DIR, 'catboost_classifier.joblib')
    
    if os.path.exists(model_path):
        model = load(model_path)
    else:
        model = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0, random_state=42)
        model.fit(X_train_smote, y_train_smote)
        dump(model, model_path)
    
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    return Calculate_Metrics("CatBoost Model", y_pred, y_test, y_score)


def train_xgboost_classifier(X_train, y_train, X_test, y_test):
    """Train and evaluate an XGBoost classifier."""

    model_path = os.path.join(MODEL_DIR, 'xgboost_classifier.joblib')

    if os.path.exists(model_path):
        model = load(model_path)
        print("Loaded XGBoost model.")
    else:
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        dump(model, model_path)
        print("Trained and saved XGBoost model.")

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    return Calculate_Metrics("XGBoost Model", y_pred, y_test, y_score)


def train_autoencoder_rf(X_train, y_train, X_test, y_test):
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    dump(scaler, os.path.join(MODEL_DIR, 'scaler_autoencoder.joblib'))

    ae_model_path = os.path.join(MODEL_DIR, 'autoencoder_model.h5')
    rf_model_path = os.path.join(MODEL_DIR, 'autoencoder_rf.joblib')

    # ------------------ Autoencoder ------------------
    if os.path.exists(ae_model_path):
        from tensorflow.keras.losses import MeanSquaredError

        autoencoder = load_model(
            ae_model_path,
            custom_objects={'mse': MeanSquaredError()}
        )

    else:
        input_dim = X_train.shape[1]
        inputs = Input(shape=(input_dim,))
        encoded = Dense(256, activation='relu')(inputs)
        bottleneck = Dense(128, activation='relu', name="bottleneck")(encoded)  # feature layer
        decoded = Dense(256, activation='relu')(bottleneck)
        outputs = Dense(input_dim, activation='linear')(decoded)

        autoencoder = Model(inputs, outputs)
        autoencoder.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        autoencoder.fit(X_train_scaled, X_train_scaled,
                        epochs=200, batch_size=32, 
                        validation_split=0.2, callbacks=[early_stop], verbose=0)

        autoencoder.save(ae_model_path)

    # Feature extractor (bottleneck layer)
    encoder = Model(inputs=autoencoder.input, 
                    outputs=autoencoder.get_layer("bottleneck").output)
    ae_features_train = encoder.predict(X_train_scaled)
    ae_features_test = encoder.predict(X_test_scaled)

    # ------------------ RF Classifier ------------------
    if os.path.exists(rf_model_path):
        rf_model = load(rf_model_path)
    else:
        rf_model = RandomForestClassifier(
            n_estimators=500, max_depth=50, random_state=42, n_jobs=-1, criterion='entropy'
        )
        rf_model.fit(ae_features_train, y_train)
        dump(rf_model, rf_model_path)

    # Predictions
    y_pred = rf_model.predict(ae_features_test)
    y_score = rf_model.predict_proba(ae_features_test)

    return Calculate_Metrics("DeepLatent-Forest", y_pred, y_test, y_score)

def predict_and_append_autoencoder_rf(df_original, df_processed, label_encoders,
                                      rf_model_filename='autoencoder_rf.joblib',
                                      ae_model_filename='autoencoder_model.h5',
                                      scaler_filename='scaler_autoencoder.joblib'):
    
    ae_model_path = os.path.join(MODEL_DIR, ae_model_filename)
    rf_model_path = os.path.join(MODEL_DIR, rf_model_filename)
    scaler_path   = os.path.join(MODEL_DIR, scaler_filename)

    if not os.path.exists(ae_model_path):
        print(" Autoencoder model file not found.")
        return None
    if not os.path.exists(rf_model_path):
        print(" Random Forest model file not found.")
        return None
    if not os.path.exists(scaler_path):
        print(" Scaler file not found.")
        return None

    # ----------------- Load Autoencoder ----------------
    from tensorflow.keras.losses import MeanSquaredError

    autoencoder = load_model(
            ae_model_path,
            custom_objects={'mse': MeanSquaredError()}
        )

    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer('bottleneck').output)

    # ----------------- Load Scaler ---------------------
    scaler = load(scaler_path)
    df_processed_np = df_processed.values.astype('float32')
    df_processed_scaled = scaler.transform(df_processed_np)

    # -------- Extract Features from Autoencoder --------
    ae_features = encoder.predict(df_processed_scaled)

    # ---------------- Load Random Forest ---------------
    rf_model = load(rf_model_path)

    # --------------------- Predict ---------------------
    y_pred = rf_model.predict(ae_features)

    # ---------------- Decode Labels --------------------
    target_col = 'Dropped Connection'
    if target_col in label_encoders:
        y_pred_decoded = label_encoders[target_col].inverse_transform(y_pred)
    else:
        y_pred_decoded = y_pred

    # ------------- Append Predictions ------------------
    df_output = df_original.copy()
    df_output['Predicted Label'] = y_pred_decoded

    output_path = os.path.join(RESULTS_DIR, "Autoencoder_RF_Predictions_Appended.csv")
    df_output.to_csv(output_path, index=False)

    return df_output



def Upload_Dataset():
    global df, labels
    text.delete('1.0', END)
    df = load_dataset()
    text.insert(END, "Dataset loaded successfully.\n\n")
    text.insert(END,str(df)+"\n\n")
    labels = df['Dropped Connection'].unique()

    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='Dropped Connection')
    plt.xlabel('Dropped Connection')
    plt.ylabel('Count')
    plt.title('Count of Class Values')
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()

def Preprocess_Dataset():
    global df, X, y, label_names, labels
    text.delete('1.0', END)    
    
    X,y,label_names  = preprocess_data(df,is_train=True)
    perform_eda(X, y)
    text.insert(END,str(X)+"\n\n")
    text.insert(END, "Preprocessing successfully completed.\n\n")
    corr = df.corr()
    labels = sorted(y.unique())
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Heatmap", fontsize=16)
    plt.show()

def Train_Test_Splitting():
    global X, y, X_train, X_test, y_test, y_train
    text.delete('1.0', END)
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    text.insert(END, "Total records found in dataset: " + str(X.shape[0]) + "\n\n")
    text.insert(END, "Dataset Train and Test Split Completed" + "\n")
    text.insert(END, "Total records found in dataset to train: " + str(X_train.shape[0]) + "\n")
    text.insert(END, "Total records found in dataset to test: " + str(X_test.shape[0]) + "\n")    
   
def existing_classifier1():
    global X_train, X_test, y_test, y_train
    text.delete('1.0', END)
    train_knn_classifier(X_train, y_train, X_test, y_test)

def existing_classifier2():
    global X_train, X_test, y_test, y_train
    train_catboost_classifier(X_train, y_train, X_test, y_test)

def existing_classifier3():
    global X_train, X_test, y_test, y_train
    train_xgboost_classifier(X_train, y_train, X_test, y_test)


def proposed_classifier3():
    global X_train, X_test, y_test, y_train
    train_autoencoder_rf(X_train, y_train, X_test, y_test)
      
def Prediction():
    global test_data, df1, final_predicted_df, label_names
    
    text.delete('1.0', END)

    # Load test dataset
    test_data = load_dataset()

    # ---- SAFETY CHECK ----
    if test_data is None:
        messagebox.showerror("Error", "No test file selected!")
        return

    if 'label_names' not in globals():
        messagebox.showerror(
            "Error",
            "Please run PREPROCESSING first before Prediction!"
        )
        return

    df1 = preprocess_data(
        test_data,
        is_train=False,
        label_encoders=label_names
    )

    final_predicted_df = predict_and_append_autoencoder_rf(
        test_data,
        df1,
        label_names
    )

    if final_predicted_df is None:
        messagebox.showerror("Error", "Prediction failed. Required models not found!")
        return

    text.insert(END, "Row-wise Prediction Results:\n\n")

    for idx, row in final_predicted_df.iterrows():
        row_text = f"Row {idx + 1} : "
        row_text += " | ".join(
            [f"{col} = {row[col]}" for col in final_predicted_df.columns]
        )
        text.insert(END, row_text + "\n")



import tkinter as tk
from tkinter import messagebox
import redis
import hashlib

# Connect to Redis
def connect_redis():
    return redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Hash password before storing in Redis for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Signup functionality
def signup(role):
    def register_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_redis()

                # Hash the password before storing
                hashed_password = hash_password(password)

                # Store the user in Redis with multiple field-value pairs
                user_key = f"user:{username}"
                if conn.exists(user_key):
                    messagebox.showerror("Error", "User already exists!")
                else:
                    # Using multiple field-value pairs in hset
                    conn.hset(user_key, "username", username)
                    conn.hset(user_key, "password", hashed_password)
                    conn.hset(user_key, "role", role)
                    messagebox.showinfo("Success", f"{role} Signup Successful!")
                    signup_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Redis Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    # Create the signup window
    signup_window = tk.Toplevel(main)
    signup_window.geometry("400x400")
    signup_window.title(f"{role} Signup")

    # Username field
    tk.Label(signup_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(signup_window)
    username_entry.pack(pady=5)
    
    # Password field
    tk.Label(signup_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(signup_window, show="*")
    password_entry.pack(pady=5)

    # Signup button
    tk.Button(signup_window, text="Signup", command=register_user).pack(pady=10)

# Login functionality
def login(role):
    def verify_user():
        username = username_entry.get()
        password = password_entry.get()

        if username and password:
            try:
                conn = connect_redis()

                # Hash the password before checking
                hashed_password = hash_password(password)

                # Check if the user exists in Redis
                user_key = f"user:{username}"
                if conn.exists(user_key):
                    stored_password = conn.hget(user_key, "password")
                    stored_role = conn.hget(user_key, "role")

                    if stored_password == hashed_password and stored_role == role:
                        messagebox.showinfo("Success", f"{role} Login Successful!")
                        login_window.destroy()
                        if role == "Admin":
                            show_admin_buttons()
                        elif role == "User":
                            show_user_buttons()
                    else:
                        messagebox.showerror("Error", "Invalid Credentials!")
                else:
                    messagebox.showerror("Error", "User not found!")
            except Exception as e:
                messagebox.showerror("Error", f"Redis Error: {e}")
        else:
            messagebox.showerror("Error", "Please enter all fields!")

    login_window = tk.Toplevel(main)
    login_window.geometry("400x300")
    login_window.title(f"{role} Login")

    tk.Label(login_window, text="Username").pack(pady=5)
    username_entry = tk.Entry(login_window)
    username_entry.pack(pady=5)

    tk.Label(login_window, text="Password").pack(pady=5)
    password_entry = tk.Entry(login_window, show="*")
    password_entry.pack(pady=5)

    tk.Button(login_window, text="Login", command=verify_user).pack(pady=10)

def show_admin_buttons():
    clear_buttons()
    tk.Button(main, text="Upload 5G Dataset", command=Upload_Dataset, font=font1).place(x=100, y=160)
    tk.Button(main, text="Preprocessing & EDA", command=Preprocess_Dataset, font=font1).place(x=320, y=160)
    tk.Button(main, text="Data Splitting", command=Train_Test_Splitting, font=font1).place(x=580, y=160)
    tk.Button(main, text="Build & Train KNN Classifier", command=existing_classifier1, font=font1).place(x=760, y=160)
    tk.Button(main, text="Build & Train CatBoost Model", command=existing_classifier2, font=font1).place(x=100, y=210)
    tk.Button(main, text="Build & Train XGB Model", command=existing_classifier3, font=font1).place(x=430, y=210)
    tk.Button(main, text="Build & Train DeepLatent-Forest Model", command=proposed_classifier3, font=font1).place(x=700, y=210)
   
def show_user_buttons():
    clear_buttons()
    tk.Button(main, text="Prediction on Test Data", command=Prediction, font=font1).place(x=650, y=200)

# Clear buttons before adding new ones
def clear_buttons():
    for widget in main.winfo_children():
        if isinstance(widget, tk.Button) and widget not in [admin_button, user_button]:
            widget.destroy()

main = tk.Tk()
screen_width = main.winfo_screenwidth()
screen_height = main.winfo_screenheight()
main.geometry(f"{screen_width}x{screen_height}")

bg_image = Image.open("background.png")  
bg_image = bg_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

canvas = Canvas(main, width=screen_width, height=screen_height)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Title
font = ('times', 18, 'bold')
title_text = "Hybrid-Autoencoder Driven ML Model for Performance Monitoring in 5G Cellular Networks"
title = tk.Label(main, text=title_text, bg='powder blue', fg='black', font=font, wraplength=screen_width - 200, justify='center')
canvas.create_window(screen_width // 2, 50, window=title)

font1 = ('times', 14, 'bold')

# Create text widget and scrollbar
text_frame = tk.Frame(main, bg='white')
text = tk.Text(text_frame, height=22, width=130, font=font1, wrap='word')
scroll = tk.Scrollbar(text_frame, command=text.yview)
text.configure(yscrollcommand=scroll.set)

text.grid(row=0, column=0, sticky='nsew')
scroll.grid(row=0, column=1, sticky='ns')
text_frame.grid_rowconfigure(0, weight=1)
text_frame.grid_columnconfigure(0, weight=1)

# Position the text_frame on the canvas, centered horizontally
canvas.create_window(screen_width // 2, 300, window=text_frame, anchor='n')


# Admin and User Buttons
font1 = ('times', 14, 'bold')

tk.Button(main, text="Admin Signup", command=lambda: signup("Admin"), font=font1, width=25, height=1, bg='salmon').place(x=50, y=100)

tk.Button(main, text="User Signup", command=lambda: signup("User"), font=font1, width=25, height=1, bg='salmon').place(x=400, y=100)

admin_button = tk.Button(main, text="Admin Login", command=lambda: login("Admin"), font=font1, width=25, height=1, bg='navajo white')
admin_button.place(x=750, y=100)

user_button = tk.Button(main, text="User Login", command=lambda: login("User"), font=font1, width=25, height=1, bg='navajo white')
user_button.place(x=1100, y=100)

main.config(bg='plum2')
main.mainloop()
