import json
import os
import matplotlib.pyplot as plt

def load_mdhistory(filename):
    with open(filename, 'r') as f:
        return json.load(f)
def plot_mdhistory(hist, model_name):
    plt.plot(hist['accuracy'], label = f'{model_name} Train')
    plt.plot(hist['val_accuracy'], label = f'{model_name} Val')

plt.figure(figsize=(10, 6))
for i in range(1, 4):
    filepath = os.path.join("DeepPCB_split", "Model results", f"model{i}_history.json")
    history = load_mdhistory(filepath)
    plot_mdhistory(history, f'Model {i}')

plt.title('Accuracy Comparison (Train vs Val)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
