import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def save_confusion_matrix(y_test, y_pred, output_path: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['setosa', 'versicolor', 'virginica'],
                yticklabels=['setosa', 'versicolor', 'virginica'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def save_feature_importance(model, feature_names, output_path: str):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def main():
    df = pd.read_csv('iris_preprocessing.csv')
    X = df.drop('species', axis=1)
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment('iris_classification_ci')

    with mlflow.start_run(run_name='random_forest_ci'):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X, y, cv=5)

        mlflow.log_param('n_estimators', model.n_estimators)
        mlflow.log_param('max_depth', model.max_depth)
        mlflow.log_param('min_samples_split', model.min_samples_split)

        mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
        mlflow.log_metric('precision_weighted', precision_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric('recall_weighted', recall_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric('f1_score_weighted', f1_score(y_test, y_pred, average='weighted'))
        mlflow.log_metric('cv_mean_accuracy', cv_scores.mean())

        save_confusion_matrix(y_test, y_pred, 'confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')

        save_feature_importance(model, X.columns.tolist(), 'feature_importance.png')
        mlflow.log_artifact('feature_importance.png')

        mlflow.sklearn.log_model(model, 'model')

        print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
        print(f'F1 Score: {f1_score(y_test, y_pred, average="weighted"):.4f}')


if __name__ == '__main__':
    main()
