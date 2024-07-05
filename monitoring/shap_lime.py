import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

data = pd.read_csv('ready_df.csv')
X = data.drop('FraudFound_P', axis=1)
y = data['FraudFound_P']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open('monitoring/model/balanced_rf_model/model.pkl', 'rb') as f:
    model = pickle.load(f)

# SHAP
explainer = shap.Explainer(model)

shap_values = explainer(X)

shap.plots.waterfall(shap_values[0,:,0], max_display=10)
shap.plots.waterfall(shap_values[0,:,1], max_display=10)

mean_0 = np.mean(np.abs(shap_values.values[:, :, 0]), axis=0)
mean_1 = np.mean(np.abs(shap_values.values[:, :, 1]), axis=0)

df = pd.DataFrame({'Not Fraud': mean_0, 'Fraud': mean_1})

fig, ax = plt.subplots(1,1,figsize=(30, 20))
df.plot.bar(ax=ax)

ax.set_ylabel('Mean SHAP', size=20)
ax.set_xticklabels(X.columns, rotation=90)
ax.legend(fontsize=30)
plt.savefig('monitoring/shap_vis/shap.png')
plt.show()

preds = model.predict(X)

new_shap_values = []
for i, pred in enumerate(preds):
    new_shap_values.append(shap_values.values[i][:, pred])

shap_values.values = np.array(new_shap_values)
shap.plots.bar(shap_values, max_display=10)

shap.plots.beeswarm(shap_values, max_display=10)

# LIME
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Not Fraud', 'Fraud'],
    mode='classification'
)

instance_idx = 0
instance = X_train.iloc[instance_idx].values.reshape(1, -1)
lime_explanation = lime_explainer.explain_instance(instance[0], model.predict_proba, num_features=10)

lime_explanation.show_in_notebook(show_all=False)
lime_explanation.save_to_file(f'monitoring/lime/lime_explanation_instance_{instance_idx}.html')
