import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
import joblib
from time import time, strftime, gmtime

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import missingno as msno

# Configuration
warnings.filterwarnings("ignore")
plt.style.use('default')
sns.set_palette("husl")


class ChurnPredictor:
    """
    A comprehensive customer churn prediction system.

    This class handles data loading, preprocessing, visualization,
    model training, and evaluation for customer churn prediction.
    """

    def __init__(self, train_path='Train.csv', test_path='Test.csv'):
        """Initialize the ChurnPredictor with data paths."""
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.models = {}
        self.results = {}

        print(f"ChurnPredictor initialized at: {datetime.datetime.now()}")
        self.start_time = time()

    def load_data(self):
        """Load training and test datasets."""
        print("Loading data...")

        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)

        # Rename labels column for clarity
        self.train_df.rename(columns={"labels": "churn"}, inplace=True)

        print(f"Training data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        print(f"Training data size: {self.train_df.size}")
        print(f"Test data size: {self.test_df.size}")

        return self

    def explore_data(self):
        """Perform comprehensive data exploration."""
        if self.train_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)

        # Display basic information
        print("\nTraining Data Head:")
        print(self.train_df.head())

        print("\nTest Data Head:")
        print(self.test_df.head())

        print("\nDescriptive Statistics:")
        print(self.train_df.describe().T)

        # Data summary
        self._display_data_summary()

        # Missing values analysis
        self._analyze_missing_values()

        return self

    def _display_data_summary(self):
        """Display comprehensive data summary."""
        print("\nDATA SUMMARY:")
        print("-" * 30)

        types_info = self.train_df.dtypes
        counts_info = self.train_df.apply(lambda x: x.count())
        uniques_info = self.train_df.apply(lambda x: x.unique().shape[0])

        summary_df = pd.concat([types_info, counts_info, uniques_info], axis=1)
        summary_df.columns = ['Data_Types', 'Non_Null_Count', 'Unique_Values']

        print(summary_df.sort_values(by='Unique_Values', ascending=False))

        print('\nData Types Distribution:')
        print(summary_df.Data_Types.value_counts())

    def _analyze_missing_values(self):
        """Analyze and visualize missing values."""
        print("\nMISSING VALUES ANALYSIS:")
        print("-" * 30)

        # Display missing values matrix
        plt.figure(figsize=(12, 6))
        msno.matrix(self.train_df)
        plt.title("Missing Values Matrix")
        plt.tight_layout()
        plt.show()

    def visualize_target_distribution(self):
        """Visualize the distribution of the target variable (churn)."""
        print("\nTARGET VARIABLE DISTRIBUTION:")
        print("-" * 40)

        churn_counts = self.train_df["churn"].value_counts()
        print(churn_counts)

        # Create pie chart
        plt.figure(figsize=(8, 6))
        sizes = churn_counts.values
        labels = ['No Churn', 'Churn']
        explode = (0, 0.1)

        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=75)
        plt.title("Customer Churn Distribution", fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

        return self

    def analyze_correlations(self):
        """Analyze and visualize feature correlations."""
        print("\nCORRELATION ANALYSIS:")
        print("-" * 25)

        # Correlation heatmap
        plt.figure(figsize=(20, 10))
        corr_matrix = self.train_df.corr()

        sns.heatmap(corr_matrix,
                    xticklabels=corr_matrix.columns.values,
                    yticklabels=corr_matrix.columns.values,
                    annot=True,
                    cmap='coolwarm',
                    center=0)
        plt.title("Feature Correlation Heatmap", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Feature correlation with target
        plt.figure(figsize=(15, 8))
        target_corr = self.train_df.corr()['churn'].sort_values(ascending=False)
        target_corr.plot(kind='bar')
        plt.title("Feature Correlation with Churn", fontsize=14, fontweight='bold')
        plt.ylabel("Correlation Coefficient")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        return self

    def visualize_features(self):
        """Create comprehensive feature visualizations."""
        print("\nFEATURE VISUALIZATION:")
        print("-" * 25)

        # Separate features by type
        features = [col for col in self.train_df.columns if col != 'churn']
        float_features = [col for col in features if self.train_df[col].dtype == 'float64']
        int_features = [col for col in features if self.train_df[col].dtype == 'int64']

        print(f"Float features: {len(float_features)}")
        print(f"Integer features: {len(int_features)}")

        # Visualize float features
        if float_features:
            self._plot_boxplots(float_features, "Float Features Box Plots")

        # Visualize integer features
        if int_features:
            self._plot_histograms(int_features)
            self._plot_boxplots(int_features, "Integer Features Box Plots")
            self._plot_feature_distributions(int_features)

        return self

    def _plot_boxplots(self, features, title):
        """Create box plots for specified features."""
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()

        for i, feature in enumerate(features):
            if i < len(axes):
                sns.boxplot(x=self.train_df[feature], ax=axes[i], palette='Set3')
                axes[i].set_title(feature)

        # Hide extra subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def _plot_histograms(self, features):
        """Create histograms for specified features."""
        for feature in features:
            plt.figure(figsize=(9, 3))
            plt.hist(self.train_df[feature], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.title(f"{feature} Distribution", fontweight='bold')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    def _plot_feature_distributions(self, features):
        """Plot feature distributions with churn analysis."""
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features))
        if n_features == 1:
            axes = [axes]

        for i, feature in enumerate(features):
            ax = sns.countplot(data=self.train_df, x=feature, hue='churn', ax=axes[i])

            # Add percentage labels
            total = len(self.train_df)
            for p in ax.patches:
                percentage = 100 * p.get_height() / total
                ax.annotate(f'{percentage:.1f}%',
                          (p.get_x() + p.get_width()/2., p.get_height()),
                          ha='center', va='bottom')

            axes[i].set_title(f'{feature} Distribution by Churn Status')

        plt.suptitle('Feature Distributions by Churn Status', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def prepare_data(self, test_size=0.3, random_state=50, use_smote=False, use_scaling=False):
        """Prepare data for model training."""
        print("\nDATA PREPARATION:")
        print("-" * 20)

        X = self.train_df.drop('churn', axis=1)
        y = self.train_df['churn']

        # Apply SMOTE if requested
        if use_smote:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE()
            X, y = smote.fit_resample(X, y)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Apply scaling if requested
        if use_scaling:
            print("Applying MinMax scaling...")
            self.scaler = MinMaxScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)

        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")

        return self

    def train_model(self, algorithm, algorithm_name, params=None):
        """Train a single model and evaluate its performance."""
        if params is None:
            params = {}

        print(f"\nTraining {algorithm_name}...")
        print("-" * 40)

        # Initialize and train model
        model = algorithm(**params)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred)
        }

        # Print results
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        # Plot confusion matrix
        self._plot_confusion_matrix(self.y_test, y_pred, algorithm_name)

        # Store model and results
        self.models[algorithm_name] = model
        self.results[algorithm_name] = metrics

        return model

    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix for model predictions."""
        cm = confusion_matrix(y_true, y_pred)
        cmd_obj = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])

        plt.figure(figsize=(6, 5))
        cmd_obj.plot(cmap='Blues')
        cmd_obj.ax_.set_title(f'{model_name} - Confusion Matrix', fontweight='bold')
        cmd_obj.ax_.set_xlabel('Predicted')
        cmd_obj.ax_.set_ylabel('Actual')
        plt.tight_layout()
        plt.show()

    def train_all_models(self):
        """Train multiple models and compare their performance."""
        print("\nTRAINING MULTIPLE MODELS:")
        print("=" * 35)

        # Define models to train
        model_configs = [
            (LogisticRegression, 'Logistic Regression', {}),
            (SVC, 'SVC Classification', {}),
            (DecisionTreeClassifier, 'Decision Tree', {}),
            (GaussianNB, 'Naive Bayes', {}),
            (AdaBoostClassifier, 'AdaBoost', {}),
            (GradientBoostingClassifier, 'Gradient Boosting', {}),
            (lgb.LGBMClassifier, 'LightGBM', {'verbose': -1}),
            (RandomForestClassifier, 'Random Forest', {'n_estimators': 100})
        ]

        # Train each model
        for algorithm, name, params in model_configs:
            try:
                self.train_model(algorithm, name, params)
            except Exception as e:
                print(f"Error training {name}: {str(e)}")

        return self

    def compare_models(self):
        """Compare all trained models and visualize results."""
        if not self.results:
            print("No models trained yet. Call train_all_models() first.")
            return

        print("\nMODEL COMPARISON:")
        print("=" * 25)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.round(4)

        print("\nModel Performance Summary:")
        print(comparison_df.sort_values('accuracy', ascending=False))

        # Plot comparison
        self._plot_model_comparison(comparison_df)

        return comparison_df

    def _plot_model_comparison(self, comparison_df):
        """Plot model performance comparison."""
        # Prepare data for plotting
        models = list(comparison_df.index)
        accuracy_scores = comparison_df['accuracy'].values

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(models)), accuracy_scores,
                      color='skyblue', edgecolor='navy', alpha=0.7)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.xlabel('Models', fontweight='bold')
        plt.ylabel('Accuracy Score', fontweight='bold')
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(models)), models, rotation=45, ha='right')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def save_best_model(self, filename='best_churn_model.pkl'):
        """Save the best performing model."""
        if not self.results:
            print("No models trained yet.")
            return

        # Find best model based on accuracy
        best_model_name = max(self.results.keys(),
                             key=lambda x: self.results[x]['accuracy'])
        best_model = self.models[best_model_name]

        # Save model
        joblib.dump(best_model, filename)

        print(f"\nBest model ({best_model_name}) saved as '{filename}'")
        print(f"Best model accuracy: {self.results[best_model_name]['accuracy']:.4f}")

        return best_model_name, best_model

    def get_execution_time(self):
        """Get total execution time."""
        finish_time = time()
        total_time = finish_time - self.start_time
        formatted_time = strftime("%H:%M:%S", gmtime(total_time))

        print(f"\nTotal execution time: {formatted_time}")
        return formatted_time


def main():
    """Main function to run the complete churn prediction pipeline."""
    print("Customer Churn Prediction Pipeline")
    print("=" * 50)

    # Initialize predictor
    predictor = ChurnPredictor('Train.csv', 'Test.csv')

    try:
        # Run the complete pipeline
        (predictor
         .load_data()
         .explore_data()
         .visualize_target_distribution()
         .analyze_correlations()
         .visualize_features()
         .prepare_data(use_smote=True, use_scaling=True)
         .train_all_models()
         .compare_models()
         .save_best_model()
         .get_execution_time())

        print("\nPipeline completed successfully!")

    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")


if __name__ == "__main__":
    main()