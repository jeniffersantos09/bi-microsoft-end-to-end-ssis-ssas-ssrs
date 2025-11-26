import pandas as pd
from sklearn.impute import SimpleImputer

class DataPreparation:
    def __init__(self, path: str):
        self.path = path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.path)
        print(f"Dataset carregado com {self.df.shape[0]} linhas e {self.df.shape[1]} colunas.")
        return self.df

    def clean_data(self):
        df = self.df.copy()

        
        df.drop_duplicates(inplace=True)

        
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        imputer = SimpleImputer(strategy="mean")
        df[num_cols] = imputer.fit_transform(df[num_cols])

       
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]

        self.df = df
        print("Limpeza concluída.")
        return df

    def save(self, output_path: str):
        self.df.to_csv(output_path, index=False)
        print(f"Dataset salvo em {output_path}")
