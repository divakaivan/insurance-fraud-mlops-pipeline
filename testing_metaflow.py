from metaflow import FlowSpec, step, current, card
from utils import utils
import pandas as pd

def convert_values_to_int_if_possible(dictionary):
    converted_dict = {}
    for key, value in dictionary.items():
        try:
            converted_dict[key] = int(value)
        except ValueError:
            converted_dict[key] = value
    return converted_dict

class MyFlow(FlowSpec):                                                                                                                                                       

    @step   
    def start(self):
        self.next(self.load_data)

    @step
    def load_data(self):
        data = pd.read_csv('ready_df.csv')
        self.data = data
        self.next(self.split_data)

    @step
    def split_data(self):
        from sklearn.model_selection import train_test_split
        
        dummies_X = self.data.drop(columns=['FraudFound_P'])
        y = self.data['FraudFound_P']
        X_train, X_test, y_train, y_test = train_test_split(dummies_X, y, test_size=0.2, random_state=42)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.next(self.train_model)

    @step
    def train_model(self):
        from imblearn.ensemble import BalancedRandomForestClassifier
        
        best_prams = utils.get_best_params()
        best_prams = convert_values_to_int_if_possible(best_prams)
        self.model = BalancedRandomForestClassifier(**best_prams)
        self.model.fit(self.X_train, self.y_train)
        self.next(self.evaluate)

    @card
    @step
    def evaluate(self):
        from sklearn.metrics import classification_report
        
        y_pred = self.model.predict(self.X_test)
        self.report = classification_report(self.y_test, y_pred)
        self.next(self.end)

    @step
    def end(self):
        print("Flow finished successfully!")

if __name__ == "__main__":
    MyFlow()
