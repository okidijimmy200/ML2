from ml_code import CTABGAN


class Inference:
    def __init__(self, raw_data, categorical_columns, integer_columns, pre_trained_paths, synthetic_data_path):
        self.raw_data = raw_data
        self.categorical_columns = categorical_columns
        self.integer_columns = integer_columns
        self.pre_trained_paths = pre_trained_paths
        self.synthetic_data_path = synthetic_data_path
    
    def init_model(self):
        self.model = CTABGAN(
            input_csv_path = self.raw_data, 
            categorical_columns = self.categorical_columns, 
            log_columns = {}, 
            mixed_columns = {}, 
            integer_columns = self.integer_columns, 
            problem_type = [],
            epochs = 0, 
            save_weights = "", 
            save_frequency = 0,
            save_path="",
            load_weights=True,
            pre_trained_paths=self.pre_trained_paths,
            )

    def get_synthetic_data(self, num_rows):
        self.model.sample_from_pretrained(num_rows, self.synthetic_data_path)