import matplotlib.pyplot as plt

def plot_revenue_by_product(product_analysis):
    plt.figure(figsize=(10,6))

    plt.bar(
        product_analysis["product_id"],
        product_analysis["total_revenue"]
    )

    plt.title("Revenue by Product")
    plt.xlabel("Product")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def plot_revenue_by_region(sales_df):
    revenue_by_region = sales_df.groupby("region")["revenue"].sum()

    plt.figure(figsize=(8,5))

    plt.bar(
        revenue_by_region.index,
        revenue_by_region.values
    )

    plt.title("Revenue by Region")
    plt.xlabel("Region")
    plt.ylabel("Revenue")
    
    plt.tight_layout()
    plt.show()


def plot_revenue_by_channel(sales_df):
    plot_revenue_by_channel = sales_df.groupby("channel")["revenue"].sum()

    plt.figure(figsize=(7,5))

    plt.bar(
        plot_revenue_by_channel.index,
        plot_revenue_by_channel.values
    )

    plt.title("Revenue by Channel")
    plt.xlabel("Channel")
    plt.ylabel("Revenue")
    
    plt.tight_layout()
    plt.show()


def plot_quantity_by_product(product_analysis):
    plt.figure(figsize=(10,6))

    plt.bar(
        product_analysis["product_id"],
        product_analysis["total_quantity"]
    )

    plt.title("Quantity Sold by Product")
    plt.xlabel("Product")
    plt.ylabel("Quantity Sold")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_profit_by_product(product_analysis):
    plt.figure(figsize=(10, 6))

    plt.bar(
        product_analysis["product_id"],
        product_analysis["total_profit"]
    )

    plt.title("Profit by Product")
    plt.xlabel("Product")
    plt.ylabel("Profit")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def plot_stock_levels(product_analysis):
    plt.figure(figsize=(10, 6))

    plt.bar(
        product_analysis["product_id"],
        product_analysis["stock"],
        label="Current Stock"
    )

    plt.plot(
        product_analysis["product_id"],
        product_analysis["reorder_threshold"],
        marker="o",
        label="Reorder Threshold"
    )

    plt.title("Stock vs Reorder Threshold")
    plt.xlabel("Product")
    plt.ylabel("Units")
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()
