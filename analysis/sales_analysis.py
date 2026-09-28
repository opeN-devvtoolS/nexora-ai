from data_loader import load_sales_data
from product_loader import load_product_data

def analyze_sales(df):

    total_revenue = df["revenue"].sum()
    total_quantity = df["quantity"].sum()
    average_price = df["selling_price"].mean()


    product_revenue = df.groupby("product_id")["revenue"].sum()
    top_products = product_revenue.sort_values(ascending=False).head(5)


    region_revenue = df.groupby("region")["revenue"].sum()
    channel_revenue = df.groupby("channel")["revenue"].sum()
    monthly_revenue = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum()



    return{
        "total_revenue": total_revenue,
        "total_quantity": total_quantity,
        "average_price": average_price,
        "top_products": top_products,
        "region_revenue": region_revenue,
        "channel_revenue": channel_revenue,
        "monthly_revenue": monthly_revenue
    }


def analyze_products(sales_df,products_df):
    merged_df = sales_df.merge(
        products_df,
        on="product_id",
        how="left",
        suffixes=("_sales","_product"))

    merged_df["profit"] = (
        merged_df["revenue"] - (merged_df["quantity"] * merged_df["unit_cost"]))

    merged_df["margin"] = (
        merged_df["profit"] / merged_df["revenue"]) * 100


    product_analysis = merged_df.groupby(["product_id", "product_name"]).agg(
        total_quantity=("quantity","sum"),
        total_revenue=("revenue","sum"),
        total_profit=("profit","sum"),
        average_margin=("margin","mean")).reset_index()

    product_analysis = product_analysis.merge(
        products_df[["product_id","stock","reorder_threshold"]],
        on="product_id",
        how="left")   

    return product_analysis



def detect_issues(product_analysis):

    issues = []

    LOW_SALES_THRESHOLD = 100
    LOW_MARGIN_THRESHOLD = 1

    for _, product in product_analysis.iterrows():

        if product["stock"] < product["reorder_threshold"]:
            issues.append({
                "product_id" : product["product_id"],
                "issue_type" : "LOW_STOCK",
                "evidence" : (
                    f"Stock {product['stock']} is below "
                    f"reorder threshold {product['reorder_threshold']}")
            })

        if product["total_quantity"] < LOW_SALES_THRESHOLD:
                issues.append({
                "product_id" : product["product_id"],
                "issue_type" : "LOW_SALES",
                "evidence" : (
                    f"Total quantity sold {product['total_quantity']} is below "
                    f"threshold {LOW_SALES_THRESHOLD}")                   
                })

        if product["average_margin"] < LOW_MARGIN_THRESHOLD:
            issues.append({
                "product_id": product["product_id"],
                "issue_type": "LOW_MARGIN",
                "evidence": (
                    f"Average margin {product['average_margin']:.2f}% "
                    f"is below threshold {LOW_MARGIN_THRESHOLD}%")     
            })


    return issues

if __name__ == "__main__":

    sales_df = load_sales_data()
    products_df = load_product_data()

    sales_results = analyze_sales(sales_df)
    
    product_analysis = analyze_products(sales_df,products_df)
    issues = detect_issues(product_analysis)
    print("\nTotal Revenue:",
          sales_results["total_revenue"])

    print("Total Quantity:",
          sales_results["total_quantity"])

    print("Average Selling Price:",
          sales_results["average_price"])

    print("\nTop 5 Products:")
    print(sales_results["top_products"])

    print("\nRevenue by Region:")
    print(sales_results["region_revenue"])

    print("\nRevenue by Channel:")
    print(sales_results["channel_revenue"])

    print("\nProduct Analysis:")
    print(product_analysis)

    print("\nDetected Issues:")

    for issue in issues:
        print(issue)


