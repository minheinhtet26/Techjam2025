import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV
# Ensure your CSV has columns: 'true_label' and 'predicted_label'
df = pd.read_csv("few_shot_prompting_evaluation.csv")

# Extract true and predicted labels
y_true = df['policy_label']
y_pred = df['predicted_label']

# Get unique labels (important for multi-class)
labels = sorted(df['policy_label'].unique())

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Optional: display as heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Optional: print detailed classification report
print(classification_report(y_true, y_pred, zero_division=0))