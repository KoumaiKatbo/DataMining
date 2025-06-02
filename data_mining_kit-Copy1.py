import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_fscore_support
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle

def pourcentage(data):
    """
    Renvoie le pourcentage de classe 1 et de classe 2
    """
    print(f"La part de classe 1 est : {(data['default payment next month'].sum()/data.shape[0]*100).round(1)} %")
    print(f"La part de classe 0 est : {((data.shape[0]-data['default payment next month'].sum())/data.shape[0]*100).round(1)} %")

class VAE(tf.keras.Model):
    """
    Implémentation VAE en TensorFlow/Keras
    """
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([
            layers.InputLayer(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(latent_dim + latent_dim),
        ])
        self.decoder = keras.Sequential([
            layers.InputLayer(shape=(latent_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid'),
        ])

    def encode(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z):
        return self.decoder(z)

    def sample(self, num_samples):
        z = tf.random.normal(shape=(num_samples, self.latent_dim))
        return self.decode(z)

def boxplot(data):
    """
    Renvoie les diagrammes en moustaches
    """
    # Couleurs personnalisées
    eblue_color = "#2274a5"
    sblue_color = "#fbd07c"
    color_pal = [eblue_color, sblue_color]
    
    # Créer les boxplots
    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
    
    # Boxplot avec outliers
    sns.boxplot(
        data=data,
        x="default payment next month",                # Classe (0 = non défaut, 1 = défaut)
        y="LIMIT_BAL",               # Montant de crédit donné (LIMIT_BAL)
        hue="default payment next month",
        palette=color_pal,
        showfliers=True,
        ax=ax[0]
    )
    
    # Boxplot sans outliers
    sns.boxplot(
        data=data,
        x="default payment next month",
        y="LIMIT_BAL",
        hue="default payment next month",
        palette=color_pal,
        showfliers=False,
        ax=ax[1]
    )
    
    # Titres
    ax[0].set_title("Montant de crédit par classe (avec outliers)")
    ax[1].set_title("Montant de crédit par classe (sans outliers)")
    
    # Noms plus clairs pour la légende
    legend_labels = ['Non défaut', 'Défaut']
    for i in range(2):
        handles, _ = ax[i].get_legend_handles_labels()
        ax[i].legend(handles, legend_labels, title="Classe")
    
    plt.tight_layout()
    plt.show()

def scatterplot(data):
    """
    """
    # Couleurs
    eblue_color = "#2274a5"
    sblue_color = "#fbd07c"
    
    # Utiliser ID comme proxy du temps, et X1 (LIMIT_BAL) comme "montant"
    f, ax = plt.subplots(figsize=(10, 4))
    
    # Données sans défaut de paiement
    sns.scatterplot(data=data[data['default payment next month'] == 0],
                    x='ID',
                    y='LIMIT_BAL',
                    color=eblue_color,
                    s=30,
                    alpha=1,
                    linewidth=0)
    
    # Données avec défaut de paiement
    sns.scatterplot(data=data[data['default payment next month'] == 1],
                    x='ID',
                    y='LIMIT_BAL',
                    color=sblue_color,
                    s=30,
                    alpha=1,
                    linewidth=0)
    
    # Ajustements esthétiques
    ax.set_title("Distribution du montant de crédit (LIMIT_BAL) selon le temps (ID)")
    plt.ylim(0, 1000000)  # Limite ajustée à l’échelle des crédits
    ax.set(xlabel="ID (proxy du temps)", ylabel="Montant du crédit (LIMIT_BAL)")
    
    plt.tight_layout()
    plt.show()

def distribution(data):
    """
    """
    # Couleurs personnalisées
    gray_color = "#2274a5"  # Classe 0
    red_color = "#fbd07c"   # Classe 1
    
    # Séparation des classes
    t0 = data[data['default payment next month'] == 0]  # Pas de défaut
    t1 = data[data['default payment next month'] == 1]  # Défaut
    
    # Toutes les colonnes sauf l'ID (et la cible si tu veux)
    var = [col for col in data.columns if col not in ['default payment next month']]
    
    num_features = len(var)
    num_cols = 4
    num_rows = num_features // num_cols + int(num_features % num_cols != 0)
    
    # Création des subplots
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(18, 4 * num_rows))
    ax = ax.flatten()  # Aplatir pour un index plus simple
    
    for idx, feature in enumerate(var):
        try:
            sns.kdeplot(
                t0[feature],
                bw_method=0.5,
                label="Classe 0 (Non défaut)",
                color=gray_color,
                fill=True,
                warn_singular=False,
                ax=ax[idx]
            )
            sns.kdeplot(
                t1[feature],
                bw_method=0.5,
                label="Classe 1 (Défaut)",
                color=red_color,
                fill=True,
                warn_singular=False,
                ax=ax[idx]
            )
            ax[idx].set_title(feature, fontsize=12)
            ax[idx].legend()
        except Exception as e:
            ax[idx].set_visible(False)  # Masquer le subplot vide en cas d'erreur
    
    # Supprimer les subplots inutilisés
    for i in range(idx + 1, len(ax)):
        ax[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

@tf.function
def compute_loss(model, x):
    """
    Fonction de perte du VAE
    """
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    mse = tf.reduce_mean(tf.keras.losses.mse(x, x_logit))
    kl_div = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))
    return mse + kl_div

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_vae(model, data, epochs=30, batch_size=32):
    """
    """
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(1000).batch(batch_size)
    optimizer = tf.keras.optimizers.Adam(1e-3)
    # Forcer la construction des variables en appelant encode avec un exemple
    _ = model.encode(data[:1])
    
    for epoch in range(epochs):
        for batch in dataset:
            train_step(model, batch, optimizer)
        print(f"Epoch {epoch+1}, Loss: {compute_loss(model, data):.4f}")

def modelisation(data):
    y = data['default payment next month']
    X = data.drop('default payment next month', axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def logistique(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

from imblearn.over_sampling import SMOTE
def smote(data):
    """
    modèle SMOTE
    """
    # 1. Séparer X et y
    X = data.drop('default payment next month', axis=1)
    y = data['default payment next month']
    
    # 2. Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
    
    # 4. Appliquer SMOTE uniquement sur l'entraînement
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # 5. Entraînement
    model = LogisticRegression(max_iter=1000)
    model.fit(X_resampled, y_resampled)
    
    # 6. Évaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

def visualisation_espace_latent(data, vae, X_test, y_test):
    """
    """
    
    mean, logvar = vae.encode(X_test.astype(np.float32))
    z = vae.reparameterize(mean, logvar).numpy()
    
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(z)
    
    plt.scatter(z_2d[:,0], z_2d[:,1], c=y_test, cmap='viridis', alpha=0.7)
    plt.colorbar()
    plt.title("Projection PCA de l’espace latent")
    plt.show()

def resultat_avec_reg_logistique(z, y_test):
    """
    """
    clf = LogisticRegression(max_iter=1000)
    clf.fit(z, y_test)
    
    y_pred = clf.predict(z)
    print(classification_report(y_test, y_pred))

def xgboost(X_train, y_train):
    """
    """
    # Modèle de base
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss')
    
    # Grille de paramètres à tester
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'scale_pos_weight': [1, sum(y_train==0) / sum(y_train==1)]  # Gérer le déséquilibre
    }
    
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid,
                               scoring='roc_auc', cv=3, verbose=2, n_jobs=-1)
    
    return grid_search
    return grid_search.fit(X_train, y_train)
    
    print(f"Best params: {grid_search.best_params_}")
    print(f"Best ROC AUC: {grid_search.best_score_}")

def resultat_avec_xgboost(X_test, y_test, grid_search):
    """
    """
    best_xgb = grid_search.best_estimator_
    y_pred = best_xgb.predict(X_test)
    y_proba = best_xgb.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

if __name__ == "__main__":
    data = pd.read_csv("default_of_credit_card_clients.csv", index_col=0)
    X_train, X_test, y_train, y_test = modelisation(data)
    logistique(X_train, X_test, y_train, y_test)