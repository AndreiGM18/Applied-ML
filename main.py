from typing import List
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

def read_data(file):
    if not os.path.exists(file):
        print("File does not exist")
        sys.exit(1)
    data = pd.read_csv(file)
    data.replace('?', np.nan, inplace=True)  # Treat "?" as a missing value
    return data

# Function to standardize numerical attributes
def standardize_numerical_attributes(data, numerical_cols):
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data

def detect_and_replace_outliers(data, label_column, binary_cols=[]):
    # Calculate IQR and determine outliers
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col not in binary_cols]
    numerical_cols = [col for col in numerical_cols if col != label_column]

    for col in numerical_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Replace outliers with NaN
        data[col] = np.where((data[col] < lower_bound) | (data[col] > upper_bound), np.nan, data[col])
    
    return data

def impute_missing_values(data, label_column, type, binary_cols=[]):
    # Identify columns with missing values
    missing_cols = data.columns[data.isnull().any()]
    
    # Separate numerical and categorical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != label_column]
    numerical_cols = [col for col in numerical_cols if col not in binary_cols]

    categorical_cols = data.select_dtypes(include=['object', 'int']).columns
    categorical_cols = [col for col in categorical_cols if col not in numerical_cols]
    categorical_cols = [col for col in categorical_cols if col != label_column]

    if type == 'univariate':
        # Univariate imputation for numerical columns
        for col in numerical_cols:
            if col in missing_cols:
                imputer_mean = SimpleImputer(strategy='mean')
                data[col] = imputer_mean.fit_transform(data[[col]])

        # Univariate imputation for categorical columns
        for col in categorical_cols:
            if col in missing_cols:
                imputer_mode = SimpleImputer(strategy='most_frequent')
                # Parse it as a DataFrame to avoid warnings
                data[col] = imputer_mode.fit_transform(data[[col]]).ravel()
    elif type == 'multivariate':
        # Separate numerical and categorical columns
        numerical_data = data[numerical_cols]
        categorical_data = data[categorical_cols]
        
        # Multivariate imputation for numerical columns
        imputer_iterative = IterativeImputer(max_iter=10, random_state=0)
        numerical_data = pd.DataFrame(imputer_iterative.fit_transform(numerical_data), columns=numerical_cols)
        
        # Multivariate imputation for categorical columns
        imputer_mode = SimpleImputer(strategy='most_frequent')
        categorical_data = pd.DataFrame(imputer_mode.fit_transform(categorical_data), columns=categorical_cols)
        
        # Combine numerical and categorical columns
        data_temp = pd.concat([numerical_data, categorical_data], axis=1)
        data = pd.concat([data_temp, data[label_column]], axis=1)

    return data

def analyze(data, file, path=""):
    train_data = read_data(file.replace("_full", "_train"))
    test_data = read_data(file.replace("_full", "_test"))

    if not os.path.exists(path):
        os.makedirs(path)

    if 'cerebrovascular_accident' in data.columns:
        label_column = 'cerebrovascular_accident'
    elif 'money' in data.columns:
        label_column = 'money'

    numerical_cols = data.select_dtypes(include=[np.number]).columns
    binary_cols = [col for col in numerical_cols if set(data[col].dropna().unique()) <= {0, 1}]

    # # Detect and replace outliers in the data
    # data = detect_and_replace_outliers(data, label_column, binary_cols)
    # train_data = detect_and_replace_outliers(train_data, label_column, binary_cols)
    # test_data = detect_and_replace_outliers(test_data, label_column, binary_cols)

    # # Impute missing values in the data
    # data = impute_missing_values(data, label_column, 'multivariate', binary_cols)
    # train_data = impute_missing_values(train_data, label_column, 'multivariate', binary_cols)
    # test_data = impute_missing_values(test_data, label_column, 'multivariate', binary_cols)

    # # Standardize numerical attributes
    # data = standardize_numerical_attributes(data, numerical_cols)
    # train_data = standardize_numerical_attributes(train_data, numerical_cols)
    # test_data = standardize_numerical_attributes(test_data, numerical_cols)

    # Identify numerical columns with only 0s and 1s
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col not in binary_cols]

    # Analysis for numerical data
    print("Numerical data analysis")
    
    # Save as a table
    print("Table")
    fig, ax = plt.subplots(figsize=(15, 6))  # Further adjust the figure size

    # Remove axis
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)

    # Create table with rounded values
    summary = data[numerical_cols].describe().round(3)
    tab = table(ax, summary, loc='center', cellLoc='center')

    # Adjust font size and scale
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)  # Set font size
    tab.scale(1.5, 1.5)   # Scale table to fit contents

    # Adjust column width to fit content
    for key, cell in tab.get_celld().items():
        cell.set_height(0.1)
        cell.set_width(0.2)

    plt.savefig(os.path.join(path, 'numerical_table.png'), bbox_inches='tight')
    plt.close()

    # Box plot for each numerical attribute
    print("Box plots")
    for column in numerical_cols:
        if column != label_column:
            fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size for each boxplot
            data.boxplot(column=column, ax=ax)
            plt.title(f'Box plot of {column}')
            plt.savefig(os.path.join(path, f"boxplot_{column}.png"))
            plt.close()

    # Discrete and ordinal attribute analysis
    print("Discrete and ordinal attribute analysis")
    discrete_ordinal_cols = data.select_dtypes(include=['object', 'int']).columns
    attribute_info = []
    for column in discrete_ordinal_cols:
        if column not in numerical_cols and column != label_column:
            num_not_null = data[column].notna().sum()
            num_unique = data[column].value_counts().shape[0]
            attribute_info.append([column, num_not_null, num_unique])
    
    # Create DataFrame from attribute information
    attribute_df = pd.DataFrame(attribute_info, columns=['Attribute', 'Examples without Missing Values', 'Unique Values'])

    # Save DataFrame as PNG and print it
    print("Discrete and ordinal attribute table")
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figure size
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=attribute_df.values, colLabels=attribute_df.columns, cellLoc='center', loc='center')
    plt.savefig(os.path.join(path, 'discrete_ordinal_table.png'), bbox_inches='tight')

    # Histograms for categorical and ordinal attribute analysis
    print("Histograms for categorical and ordinal attributes")
    cat_ordinal_cols = data.select_dtypes(include=['object', 'int']).columns
    for column in cat_ordinal_cols:
        if column not in numerical_cols and column != label_column:
            # Plot histogram
            fig, ax = plt.subplots(figsize=(8, 6))
            data[column].value_counts().plot(kind='bar', ax=ax, width=0.8)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(path, f"histogram_{column}.png"))
            plt.close()
    
    # Frequency of each label (class)
    for dataset in [data, train_data, test_data]:
        if label_column:
            label = 'full' if dataset.equals(data) else 'train' if dataset.equals(train_data) else 'test'
            print(f"Frequency of each label (class) for column: {label_column}")
            plt.figure(figsize=(8, 6))
            sns.countplot(data=dataset, x=label_column)
            plt.title(f'Class Distribution for {label_column} ({label} dataset)')
            plt.xlabel('Class')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(path, f'class_distribution_{label_column}_{label}.png'))
            plt.close()

    # Correlation analysis for numerical attributes
    print("Correlation analysis for numerical attributes")
    corr_matrix = data[numerical_cols].corr(method='pearson')
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Matrix for Numerical Attributes')
    plt.savefig(os.path.join(path, 'numerical_correlation_matrix.png'), bbox_inches='tight')
    plt.close()

    # Correlation analysis for categorical attributes using Chi-Square test
    print("Correlation analysis for categorical attributes")
    cat_columns = data.select_dtypes(include=['object', 'int']).columns
    cat_columns = [col for col in cat_columns if col not in label_column]
    cat_columns = [col for col in cat_columns if col not in numerical_cols]
    chi2_results = pd.DataFrame(index=cat_columns, columns=cat_columns)

    for col1 in cat_columns:
        for col2 in cat_columns:
            if col1 != col2:
                contingency_table = pd.crosstab(data[col1], data[col2])
                chi2, p, dof, ex = chi2_contingency(contingency_table)
                chi2_results.loc[col1, col2] = round(p, 3)
            else:
                chi2_results.loc[col1, col2] = np.nan
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(chi2_results.astype(float), annot=True, fmt='.3f', cmap='viridis')
    plt.title('Chi-Square Test P-Value Matrix for Categorical Attributes')
    plt.savefig(os.path.join(path, 'categorical_correlation_matrix.png'), bbox_inches='tight')
    plt.close()

def train(data, file, approach='logistic_regression', path=""):
    train_data = read_data(file.replace("_full", "_train"))
    test_data = read_data(file.replace("_full", "_test"))

    if not os.path.exists(path):
        os.makedirs(path)

    if 'cerebrovascular_accident' in data.columns:
        label_column = 'cerebrovascular_accident'
    elif 'money' in data.columns:
        label_column = 'money'

    numerical_cols = data.select_dtypes(include=[np.number]).columns
    binary_cols = [col for col in numerical_cols if set(data[col].dropna().unique()) <= {0, 1}]
    numerical_cols = [col for col in numerical_cols if col not in binary_cols]
    numerical_cols = [col for col in numerical_cols if col != label_column]

    if approach == 'logistic_regression':
        # Detecting and replacing outliers
        data = detect_and_replace_outliers(data, label_column, binary_cols)
        train_data = detect_and_replace_outliers(train_data, label_column, binary_cols)
        test_data = detect_and_replace_outliers(test_data, label_column, binary_cols)

        # Imputing missing values
        data = impute_missing_values(data, label_column, 'multivariate', binary_cols)
        train_data = impute_missing_values(train_data, label_column, 'multivariate', binary_cols)
        test_data = impute_missing_values(test_data, label_column, 'multivariate', binary_cols)

        # Standardizing numerical attributes
        data = standardize_numerical_attributes(data, numerical_cols)
        train_data = standardize_numerical_attributes(train_data, numerical_cols)
        test_data = standardize_numerical_attributes(test_data, numerical_cols)

        # Codifying categorical attributes
        categorical_cols = data.select_dtypes(include=['object', 'int']).columns
        categorical_cols = [col for col in categorical_cols if col != label_column]
        categorical_cols = [col for col in categorical_cols if col not in numerical_cols]
        
        data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        train_data_encoded = pd.get_dummies(train_data, columns=categorical_cols, drop_first=True)
        test_data_encoded = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)

        # Ensure that the same columns are present in the training and test sets
        train_data_encoded, test_data_encoded = train_data_encoded.align(test_data_encoded, join='left', axis=1, fill_value=0)
        train_data_encoded, data_encoded = train_data_encoded.align(data_encoded, join='left', axis=1, fill_value=0)

        # Codifying the label column
        le = LabelEncoder()

        data_encoded[label_column] = le.fit_transform(data_encoded[label_column])
        train_data_encoded[label_column] = le.transform(train_data_encoded[label_column])
        test_data_encoded[label_column] = le.transform(test_data_encoded[label_column])

        # Handling class imbalance with SMOTE
        smote = SMOTE(random_state=0)
        X_train, y_train = smote.fit_resample(train_data_encoded.drop(columns=[label_column]), train_data_encoded[label_column])

        X_test = test_data_encoded.drop(columns=[label_column])
        y_test = test_data_encoded[label_column]

        # Training the model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluating the model
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Confusion Matrices
        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_test = confusion_matrix(y_test, y_pred_test)

        # Print results in path
        with open(os.path.join(path, 'results.txt'), 'w') as f:
            f.write("Train set\n")
            f.write(f"Accuracy: {accuracy_score(y_train, y_pred_train)}\n")
            f.write(classification_report(y_train, y_pred_train))

            f.write("\nTest set\n")
            f.write(f"Accuracy: {accuracy_score(y_test, y_pred_test)}\n")
            f.write(classification_report(y_test, y_pred_test))

            f.write("\nTrain Confusion Matrix:\n")
            f.write(np.array2string(cm_train))
            f.write("\nTest Confusion Matrix:\n")
            f.write(np.array2string(cm_test))

    elif approach == 'MLP':
        # Detecting and replacing outliers
        data = detect_and_replace_outliers(data, label_column, binary_cols)
        train_data = detect_and_replace_outliers(train_data, label_column, binary_cols)
        test_data = detect_and_replace_outliers(test_data, label_column, binary_cols)

        # Imputing missing values
        data = impute_missing_values(data, label_column, 'multivariate', binary_cols)
        train_data = impute_missing_values(train_data, label_column, 'multivariate', binary_cols)
        test_data = impute_missing_values(test_data, label_column, 'multivariate', binary_cols)

        # Standardizing numerical attributes
        data = standardize_numerical_attributes(data, numerical_cols)
        train_data = standardize_numerical_attributes(train_data, numerical_cols)
        test_data = standardize_numerical_attributes(test_data, numerical_cols)

        # Codifying categorical attributes
        categorical_cols = data.select_dtypes(include=['object', 'int']).columns
        categorical_cols = [col for col in categorical_cols if col != label_column]
        categorical_cols = [col for col in categorical_cols if col not in numerical_cols]
        
        data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        train_data_encoded = pd.get_dummies(train_data, columns=categorical_cols, drop_first=True)
        test_data_encoded = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)

        # Ensure that the same columns are present in the training and test sets
        train_data_encoded, test_data_encoded = train_data_encoded.align(test_data_encoded, join='left', axis=1, fill_value=0)
        train_data_encoded, data_encoded = train_data_encoded.align(data_encoded, join='left', axis=1, fill_value=0)

        # Codifying the label column
        le = LabelEncoder()

        data_encoded[label_column] = le.fit_transform(data_encoded[label_column])
        train_data_encoded[label_column] = le.transform(train_data_encoded[label_column])
        test_data_encoded[label_column] = le.transform(test_data_encoded[label_column])

        # Handling class imbalance with SMOTE
        smote = SMOTE(random_state=0)
        X_train, y_train = smote.fit_resample(train_data_encoded.drop(columns=[label_column]), train_data_encoded[label_column])

        X_test = test_data_encoded.drop(columns=[label_column])
        y_test = test_data_encoded[label_column]

        # Training the MLP model
        model = MLPClassifier(random_state=0, max_iter=1000, hidden_layer_sizes=(64, 32), activation='relu', solver='adam', learning_rate_init=0.001, alpha=0.0001, batch_size=32)

        # Tracking the loss and accuracy for each epoch
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        for epoch in range(model.max_iter):
            model.partial_fit(X_train, y_train, classes=np.unique(y_train))
            
            # Calculate training loss and accuracy
            y_pred_train_prob = model.predict_proba(X_train)
            y_pred_train = np.argmax(y_pred_train_prob, axis=1)
            train_loss.append(log_loss(y_train, y_pred_train_prob))
            train_acc.append(accuracy_score(y_train, y_pred_train))

            # Calculate test loss and accuracy
            y_pred_test_prob = model.predict_proba(X_test)
            y_pred_test = np.argmax(y_pred_test_prob, axis=1)
            test_loss.append(log_loss(y_test, y_pred_test_prob))
            test_acc.append(accuracy_score(y_test, y_pred_test))

            # Early stopping condition
            if epoch > 10 and abs(test_loss[-1] - test_loss[-2]) < 0.001:
                break

        # Evaluating the model
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Print results in path
        with open(os.path.join(path, f'results_{approach}.txt'), 'w') as f:
            f.write("Train set\n")
            f.write(f"Accuracy: {accuracy_score(y_train, y_pred_train)}\n")
            f.write(classification_report(y_train, y_pred_train))

            f.write("\nTest set\n")
            f.write(f"Accuracy: {accuracy_score(y_test, y_pred_test)}\n")
            f.write(classification_report(y_test, y_pred_test))

            # Confusion Matrices
            cm_train = confusion_matrix(y_train, y_pred_train)
            cm_test = confusion_matrix(y_test, y_pred_test)

            f.write("\nTrain Confusion Matrix:\n")
            f.write(np.array2string(cm_train))
            f.write("\nTest Confusion Matrix:\n")
            f.write(np.array2string(cm_test))

        # Plotting the training and test loss curves
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Train Loss')
        plt.plot(test_loss, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Test Loss')

        # Plotting the training and test accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(test_acc, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Test Accuracy')

        plt.tight_layout()
        plt.savefig(os.path.join(path, 'learning_curves.png'))
        plt.close()


def logistic(x):
    return 1 / (1 + np.exp(-x.astype(float)))

def nll(Y, T):
    N = T.shape[0]

    return -np.sum(T * np.log(Y) + (1 - T) * np.log(1 - Y)) / N

def accuracy(Y, T):
    N = Y.shape[0]

    return np.sum((Y >= 0.5) == T)/N

def predict_logistic(X, w):
    N = X.shape[0]
    return logistic(np.dot(X, w))

def train_and_eval_logistic(X, X_train, T_train, X_test, T_test, lr=.01, epochs_no=100):
    #  <3.2> : Antrenati modelul logistic (ponderile W), executand epochs_no pasi din algoritmul de gradient descent
    (N, D) = X.shape
    
    # Initializare ponderi
    w = np.random.randn(D)
    
    train_acc, test_acc = [], []
    train_nll, test_nll = [], []

    for epoch in range(epochs_no):
        # 1. Obtineti Y_train si Y_test folosind functia predict_logistic
        Y_train = predict_logistic(X_train, w)
        Y_test = predict_logistic(X_test, w)
        # 2. Adaugati acuratetea si negative log likelihood-ul pentru setul de antrenare si de testare 
        #    la fiecare pas; utilizati functiile accuracy si nll definite anterior
        train_acc.append(accuracy(Y_train, T_train))
        test_acc.append(accuracy(Y_test, T_test))
        
        train_nll.append(nll(Y_train, T_train))
        test_nll.append(nll(Y_test, T_test))

        # 3. Actualizati ponderile w folosind regula de actualizare a gradientului
        gradient = np.dot(X_train.T, Y_train - T_train) / N
        w -= lr * gradient.astype(float)      

    return w, train_nll, test_nll, train_acc, test_acc

class Layer:

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def update(self, *args, **kwargs):
        pass  # If a layer has no parameters, then this function does nothing

class FeedForwardNetwork:
    
    def __init__(self, layers: List[Layer]):
        self.layers = layers
        
    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self._inputs = []
        for layer in self.layers:
            if train:
                self._inputs.append(x)
            x = layer.forward(x)
        return x
    
    def backward(self, dy: np.ndarray) -> np.ndarray:
        #  <0> : Calculati gradientul cu fiecare strat
        # Pasi:
        #   - iterati in ordine inversa prin straturile retelei si apelati pentru fiecare dintre ele metoda backward
        #   - folositi self._inputs salvate la fiecare pas din forward pentru a calcula gradientul cu respectivul strat
        #   - transmiteti mai departe valoarea returnata de metoda backward catre urmatorul strat
        #   - incepeti cu gradientul fata de output (dy, primit ca argument).
        # Initialize the gradient of the last layer with respect to its input
        for i in reversed(range(len(self.layers))):
            dy = self.layers[i].backward(self._inputs[i], dy)
        del self._inputs

        return dy
    
    def update(self, *args, **kwargs):
        for layer in self.layers:
            layer.update(*args, **kwargs)

class Linear(Layer):
    
    def __init__(self, insize: int, outsize: int) -> None:
        bound = np.sqrt(6. / insize)
        self.weight = np.random.uniform(-bound, bound, (insize, outsize))
        self.bias = np.zeros((outsize,))
        
        self.dweight = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)
   
    def forward(self, x: np.ndarray) -> np.ndarray:
        #  <1>: calculați ieșirea unui strat liniar
        # x - este o matrice numpy B x M, unde 
        #    B - dimensiunea batchului, 
        #    M - dimensiunea caracteristicilor de intrare (insize)
        # Sugestie: folosiți înmulțirea matricială numpy pentru a implementa propagarea înainte într-o singură trecere
        # pentru toate exemplele din batch
        
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        #  <2> : calculați dweight, dbias și returnați dx
        # x - este o matrice numpy B x M, unde 
        #     B - dimensiunea batchului, 
        #     M - dimensiunea caracteristicilor (features) de intrare (insize)
        # dy - este o matrice numpy B x N, unde 
        #     B - dimensiunea batchului, 
        #     N - dimensiunea caracteristicilor (features) de ieșire (outsize)
        # Sugestie: folosiți înmulțirea matricială numpy pentru a implementa propagarea înapoi într-o singură trecere 
        #       pentru self.dweight
        # Sugestie: folosiți numpy.sum pentru a implementa propagarea înapoi într-o singură trecere pentru self.dbias

        self.dweight = np.dot(x.T, dy)

        self.dbias = np.sum(dy, axis=0)

        dx = np.dot(dy, self.weight.T)
        
        return dx
    
    def update(self, mode='SGD', lr=0.001, mu=0.9):
        if mode == 'SGD':
            self.weight -= lr * self.dweight.astype(float)
            self.bias -= lr * self.dbias.astype(float)
        else:
            raise ValueError('mode should be SGD, not ' + str(mode))

class ReLU(Layer):
    
    def __init__(self) -> None:
        pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        #  <3> : Calculați ieșirea unei unități liniare rectificate
        return np.maximum(x, 0)
    
    def backward(self, x: np.ndarray, dy: np.ndarray) -> np.ndarray:
        #  <4> : Calculați gradientul față de x
        # x - este o matrice numpy B x M, unde B - dimensiunea batchului, M - dimensiunea caracteristicilor
        # Sugestie: utilizați indexarea logică numpy pentru a determina unde intrarea (x) este negativă
        #       și faceți gradientul 0 pentru acele exemple

        return dy * (x > 0).astype(np.float32)

class CrossEntropy:
    
    def __init__(self):
        pass
    
    def softmax(self, x):
        exps = np.exp(x.astype(float))
        return exps / np.sum(exps,axis = 1).reshape(-1,1)

    def forward(self, y: np.ndarray, t: np.ndarray) -> float:
        #  <5> : Calculați probabilitatea logaritmică negativă
        # y - matrice numpy (B, K), unde B - dimensiunea batch-ului, K - numărul de clase (numărul de logaritmi)
        # t - vector numpy (B, ), unde B - dimensiunea batch-ului, care indică clasa corectă
        # Pasi: 
        #   - folositi softmax() pe intrari pentru a transforma logits (y) in probabilitati
        #   - selectati probabilitatile care corespund clasei reale (t)
        #   - calculati -log() peste probabilitati
        #   - impartiti la batch size pentru a calcula valoarea medie peste toate exemplele din batch

        softmax_y = self.softmax(y)
        return -np.mean(np.log(softmax_y[np.arange(len(softmax_y)), t]))
    
    def backward(self, y: np.ndarray, t: np.ndarray) -> np.ndarray:
        #  <6> : Calculati dl/dy
        # Pasi: 
        #   - calculati softmax(y) pentru a determina probabilitatea ca fiecare element sa apartina clasei i
        #   - ajustati gradientii pentru clasa corecta: aplicati scaderea dL/dy_i = pi - delta_ti conform formulelor de mai sus
        #   - impartiti la batch size pentru a calcula valoarea medie peste toate exemplele din batch

        softmax_y = self.softmax(y)
        grad = softmax_y.copy()
        grad[np.arange(len(softmax_y)), t] -= 1
        grad /= len(softmax_y)
        return grad

def accuracyMLP(y: np.ndarray, t: np.ndarray) -> float:
    #  <7> : Calculati acuratetea
    # Pasi: 
    # - folosiți np.argmax() pentru a afla predictiile retelei
    # - folositi np.sum() pentru a numara cate sunt corecte comparand cu ground truth (t)
    # - impartiti la batch size pentru a calcula valoarea medie peste toate exemplele din batch
    return np.mean(np.argmax(y, axis=1) == t)

def precision(y_pred, y_true):
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0

def recall(y_pred, y_true):
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives != 0 else 0

def f1_score(y_pred, y_true):
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0

# Include these new metrics in your evaluation
def evaluate_metrics(y_pred, y_true):
    acc = accuracy(y_pred, y_true)
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    f1 = f1_score(y_pred, y_true)
    return acc, prec, rec, f1

def my_confusion_matrix(y_true, y_pred):
    unique_classes = np.unique(y_true)
    cm = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    for i, j in zip(y_true, y_pred):
        cm[i, j] += 1
    return cm

def my_train(data, file, approach='logistic_regression', path=""):
    train_data = read_data(file.replace("_full", "_train"))
    test_data = read_data(file.replace("_full", "_test"))

    if not os.path.exists(path):
        os.makedirs(path)

    if 'cerebrovascular_accident' in data.columns:
        label_column = 'cerebrovascular_accident'
    elif 'money' in data.columns:
        label_column = 'money'

    numerical_cols = data.select_dtypes(include=[np.number]).columns
    binary_cols = [col for col in numerical_cols if set(data[col].dropna().unique()) <= {0, 1}]
    numerical_cols = [col for col in numerical_cols if col not in binary_cols]
    numerical_cols = [col for col in numerical_cols if col != label_column]

    if approach == 'logistic_regression':
        # Detecting and replacing outliers
        data = detect_and_replace_outliers(data, label_column, binary_cols)
        train_data = detect_and_replace_outliers(train_data, label_column, binary_cols)
        test_data = detect_and_replace_outliers(test_data, label_column, binary_cols)

        # Imputing missing values
        data = impute_missing_values(data, label_column, 'multivariate', binary_cols)
        train_data = impute_missing_values(train_data, label_column, 'multivariate', binary_cols)
        test_data = impute_missing_values(test_data, label_column, 'multivariate', binary_cols)

        # Standardizing numerical attributes
        data = standardize_numerical_attributes(data, numerical_cols)
        train_data = standardize_numerical_attributes(train_data, numerical_cols)
        test_data = standardize_numerical_attributes(test_data, numerical_cols)

        # Codifying categorical attributes
        categorical_cols = data.select_dtypes(include=['object', 'int']).columns
        categorical_cols = [col for col in categorical_cols if col != label_column]
        categorical_cols = [col for col in categorical_cols if col not in numerical_cols]
        
        data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        train_data_encoded = pd.get_dummies(train_data, columns=categorical_cols, drop_first=True)
        test_data_encoded = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)

        # Ensure that the same columns are present in the training and test sets
        train_data_encoded, test_data_encoded = train_data_encoded.align(test_data_encoded, join='left', axis=1, fill_value=0)
        train_data_encoded, data_encoded = train_data_encoded.align(data_encoded, join='left', axis=1, fill_value=0)

        # Codifying the label column
        le = LabelEncoder()

        data_encoded[label_column] = le.fit_transform(data_encoded[label_column])
        train_data_encoded[label_column] = le.transform(train_data_encoded[label_column])
        test_data_encoded[label_column] = le.transform(test_data_encoded[label_column])

        # Handling class imbalance with SMOTE
        smote = SMOTE(random_state=0)
        X_train, y_train = smote.fit_resample(train_data_encoded.drop(columns=[label_column]), train_data_encoded[label_column])

        X_test = test_data_encoded.drop(columns=[label_column])
        y_test = test_data_encoded[label_column]

        # Training the model
        w, train_nll, test_nll, train_acc, test_acc = train_and_eval_logistic(data_encoded.drop(columns=[label_column]).values, X_train, y_train, X_test, y_test)

        # Evaluate metrics on train set
        y_train_pred = predict_logistic(X_train, w) >= 0.5
        train_acc, train_prec, train_rec, train_f1 = evaluate_metrics(y_train_pred, y_train)
        train_cm = confusion_matrix(y_train, y_train_pred)

        # Evaluate metrics on test set
        y_test_pred = predict_logistic(X_test, w) >= 0.5
        test_acc, test_prec, test_rec, test_f1 = evaluate_metrics(y_test_pred, y_test)
        test_cm = my_confusion_matrix(y_test, y_test_pred)

        # Print results in path
        with open(os.path.join(path, 'my_results.txt'), 'w') as f:
            f.write("Train set\n")
            f.write(f"Accuracy: {train_acc}\n")
            f.write(f"Precision: {train_prec}\n")
            f.write(f"Recall: {train_rec}\n")
            f.write(f"F1 Score: {train_f1}\n")
            f.write(f"Negative Log Likelihood: {train_nll[-1]}\n")
            f.write(f"Confusion Matrix:\n{train_cm}\n")

            f.write("\nTest set\n")
            f.write(f"Accuracy: {test_acc}\n")
            f.write(f"Precision: {test_prec}\n")
            f.write(f"Recall: {test_rec}\n")
            f.write(f"F1 Score: {test_f1}\n")
            f.write(f"Negative Log Likelihood: {test_nll[-1]}\n")
            f.write(f"Confusion Matrix:\n{test_cm}\n")
    
    elif approach == 'MLP':
        # Detecting and replacing outliers
        data = detect_and_replace_outliers(data, label_column, binary_cols)
        train_data = detect_and_replace_outliers(train_data, label_column, binary_cols)
        test_data = detect_and_replace_outliers(test_data, label_column, binary_cols)

        # Imputing missing values
        data = impute_missing_values(data, label_column, 'multivariate', binary_cols)
        train_data = impute_missing_values(train_data, label_column, 'multivariate', binary_cols)
        test_data = impute_missing_values(test_data, label_column, 'multivariate', binary_cols)

        # Standardizing numerical attributes
        data = standardize_numerical_attributes(data, numerical_cols)
        train_data = standardize_numerical_attributes(train_data, numerical_cols)
        test_data = standardize_numerical_attributes(test_data, numerical_cols)

        # Codifying categorical attributes
        categorical_cols = data.select_dtypes(include=['object', 'int']).columns
        categorical_cols = [col for col in categorical_cols if col != label_column]
        categorical_cols = [col for col in categorical_cols if col not in numerical_cols]
        
        data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
        train_data_encoded = pd.get_dummies(train_data, columns=categorical_cols, drop_first=True)
        test_data_encoded = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)

        # Ensure that the same columns are present in the training and test sets
        train_data_encoded, test_data_encoded = train_data_encoded.align(test_data_encoded, join='left', axis=1, fill_value=0)
        train_data_encoded, data_encoded = train_data_encoded.align(data_encoded, join='left', axis=1, fill_value=0)

        # Codifying the label column
        le = LabelEncoder()

        data_encoded[label_column] = le.fit_transform(data_encoded[label_column])
        train_data_encoded[label_column] = le.transform(train_data_encoded[label_column])
        test_data_encoded[label_column] = le.transform(test_data_encoded[label_column])

        # Handling class imbalance with SMOTE
        smote = SMOTE(random_state=0)
        X_train, y_train = smote.fit_resample(train_data_encoded.drop(columns=[label_column]), train_data_encoded[label_column])

        X_test = test_data_encoded.drop(columns=[label_column])
        y_test = test_data_encoded[label_column]

        BATCH_SIZE = 32
        HIDDEN_UNITS = 64
        EPOCHS_NO = 20

        optimize_args = {'mode': 'SGD', 'lr': .005}

        net = FeedForwardNetwork([
            Linear(X_train.shape[1], HIDDEN_UNITS),
            ReLU(),
            Linear(HIDDEN_UNITS, 2)
        ])
        cost_function = CrossEntropy()

        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []

        for epoch in range(EPOCHS_NO):
            epoch_train_loss = 0
            epoch_train_accuracy = 0
            for b_no, idx in enumerate(range(0, len(X_train), BATCH_SIZE)):
                # Prepare the next batch
                x = X_train.iloc[idx:idx + BATCH_SIZE].values.reshape(-1, X_train.shape[1])
                t = y_train.iloc[idx:idx + BATCH_SIZE].values
                
                # Forward pass
                y = net.forward(x)
                loss = cost_function.forward(y, t)
                
                # Backward pass
                dy = cost_function.backward(y, t)
                net.backward(dy)
                
                # Update parameters
                net.update(**optimize_args)
                
                # Accumulate loss and accuracy for the current batch
                epoch_train_loss += loss
                epoch_train_accuracy += accuracyMLP(y, t)

            # Calculate average loss and accuracy for the epoch
            epoch_train_loss /= (len(X_train) / BATCH_SIZE)
            epoch_train_accuracy /= (len(X_train) / BATCH_SIZE)
            
            # Append the metrics for the epoch
            train_loss.append(epoch_train_loss)
            train_accuracy.append(epoch_train_accuracy)
            
            # Evaluate on the test set after each epoch
            y_test_pred = net.forward(X_test.values.reshape(-1, X_test.shape[1]), train=False)
            test_loss.append(cost_function.forward(y_test_pred, y_test.values))
            test_accuracy.append(accuracyMLP(y_test_pred, y_test.values))

        # Plot the curves
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, EPOCHS_NO + 1), train_loss, label='Train Loss')
        plt.plot(range(1, EPOCHS_NO + 1), test_loss, label='Test Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(path, 'loss_curve.png'))
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, EPOCHS_NO + 1), train_accuracy, label='Train Accuracy')
        plt.plot(range(1, EPOCHS_NO + 1), test_accuracy, label='Test Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(path, 'accuracy_curve.png'))
        plt.close()

        # Print results in path
        with open(os.path.join(path, 'my_results.txt'), 'w') as f:
            for i in range(EPOCHS_NO):
                f.write(f'Epoch {i+1}: Train Loss={train_loss[i]}, Train Accuracy={train_accuracy[i]}, Test Loss={test_loss[i]}, Test Accuracy={test_accuracy[i]}\n')

            # Calculate additional metrics for the training set
            y_train_pred = net.forward(X_train.values.reshape(-1, X_train.shape[1]), train=False)
            train_prec, train_rec, train_f1 = precision(np.argmax(y_train_pred, axis=1), y_train), recall(np.argmax(y_train_pred, axis=1), y_train), f1_score(np.argmax(y_train_pred, axis=1), y_train)
            train_acc = accuracyMLP(y_train_pred, y_train)
            f.write(f'Train Accuracy: {train_acc * 100:3.2f}% '
                    f'| Train Prec: {train_prec * 100:3.2f}% '
                    f'| Train Rec: {train_rec * 100:3.2f}% '
                    f'| Train F1: {train_f1 * 100:3.2f}%\n')
            
            # Calculate confusion matrix for train set
            y_train_pred_class = np.argmax(y_train_pred, axis=1)
            tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred_class).ravel()
            f.write("Train Confusion Matrix:\n")
            f.write(f"      Predicted Negative   Predicted Positive\n")
            f.write(f"Actual Negative    {tn}                {fp}\n")
            f.write(f"Actual Positive    {fn}                {tp}\n")

            # Evaluate on the test set
            y_test_pred = net.forward(X_test.values.reshape(-1, X_test.shape[1]), train=False)
            test_nll = cost_function.forward(y_test_pred, y_test.values)
            f.write(f'| Test NLL: {test_nll:6.3f} '
                    f'| Test Acc: {accuracyMLP(y_test_pred, y_test.values) * 100:3.2f}% ')
            
            # Calculate additional metrics
            y_test_pred_class = np.argmax(y_test_pred, axis=1)
            test_prec, test_rec, test_f1 = precision(y_test_pred_class, y_test), recall(y_test_pred_class, y_test), f1_score(y_test_pred_class, y_test)
            f.write(f'| Test Prec: {test_prec * 100:3.2f}% '
                    f'| Test Rec: {test_rec * 100:3.2f}% '
                    f'| Test F1: {test_f1 * 100:3.2f}%\n')

            # Calculate confusion matrix for test set
            tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_class).ravel()
            f.write("Test Confusion Matrix:\n")
            f.write(f"      Predicted Negative   Predicted Positive\n")
            f.write(f"Actual Negative    {tn}                {fp}\n")
            f.write(f"Actual Positive    {fn}                {tp}\n")

# Main function
if __name__ == "__main__":
    salary_file = "tema2_SalaryPrediction/SalaryPrediction_full.csv"
    salary_data = read_data(salary_file)
    # analyze(salary_data, salary_file, path="SalaryPrediction_analysis")
    # train(salary_data, salary_file, approach='logistic_regression', path="SalaryPrediction_scikit_learn")
    # my_train(salary_data, salary_file, approach='MLP', path="SalaryPrediction_my_MLP")

    avc_file = "tema2_AVC/AVC_full.csv"
    avc_data = read_data(avc_file)
    # analyze(avc_data, avc_file, path="AVC_analysis")
    # train(avc_data, avc_file, approach='logistic_regression', path="AVC_scikit_learn")
    my_train(avc_data, avc_file, approach='MLP', path="AVC_my_MLP")
