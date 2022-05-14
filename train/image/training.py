from ml_code import CTABGAN
import torch

class Training:
    def __init__(self, input_csv_path, categorical_columns, log_columns, mixed_columns, integer_columns, problem_type, epochs, save_weights, save_frequency, save_path, model_name):
        self.input_csv_path = input_csv_path
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        self.epochs = epochs
        self.save_weights = save_weights
        self.save_frequency = save_frequency
        self.save_path = save_path
        self.model_name = model_name
    
    def init_model(self):
        print("epochs in training class", self.epochs)
        self.model = CTABGAN(
            input_csv_path = self.input_csv_path, 
            categorical_columns = self.categorical_columns, 
            log_columns = self.log_columns, 
            mixed_columns = self.mixed_columns, 
            integer_columns = self.integer_columns, 
            problem_type = self.problem_type, 
            epochs = self.epochs, 
            save_weights = self.save_weights, 
            save_frequency = self.save_frequency,
            save_path=self.save_path,
            )
    
    def train(self):
        self.model.fit()



