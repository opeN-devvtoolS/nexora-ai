from data_loader import load_sales_data
from product_loader import load_product_data
from sales_analysis import analyze_products

from visualizations import (
    plot_revenue_by_product, 
    plot_revenue_by_region,
    plot_revenue_by_channel,
    plot_quantity_by_product,
    plot_profit_by_product,
    plot_stock_levels
)


sales_df = load_sales_data()
products_df = load_product_data()

product_analysis = analyze_products(
    sales_df,
    products_df
)

# plot_revenue_by_product(product_analysis)
# plot_revenue_by_region(sales_df)
# plot_revenue_by_channel(sales_df)
# plot_quantity_by_product(product_analysis)
# plot_profit_by_product(product_analysis)
plot_stock_levels(product_analysis)