import pandas as pd

def load_product_data():
    df = pd.read_csv("data/products.csv")


    df = df.dropna()
    df = df.drop_duplicates()

    return df



if __name__ == "__main__":

    df = load_product_data()

    print(df.head())
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Data types:")
    print(df.dtypes)
    print("Missing values:")
    print(df.isnull().sum())