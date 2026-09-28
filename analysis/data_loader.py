import pandas as pd

def load_sales_data():
    df = pd.read_csv("data/sales.csv")

    df["date"] = pd.to_datetime(df["date"])
    df["revenue"] = df["quantity"]*df["selling_price"] * (1-df["discount"])
    df["is_return"] = df["quantity"] < 0
    
    df = df.dropna()

    df = df.drop_duplicates()

    return df



if __name__ == "__main__":

    df = load_sales_data()
    print(df.head())
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Data types:")
    print(df.dtypes)
    print("Missing values:")
    print(df.isnull().sum())
    print("Returns:", df["is_return"].sum())