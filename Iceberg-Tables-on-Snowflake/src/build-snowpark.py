# The Snowpark package is required for Python Worksheets. 
# You can add more packages by selecting them using the Packages control and then importing them.

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col

def main(session: snowpark.Session): 

    menu_df = session.table('menu')
    orders_df = session.table('orders_iceberg').drop_duplicates()

    agg_df = session.sql("""
                            SELECT  
                                b.truck_brand_name, 
                                sum(b.cost_of_goods_usd) as total_cost, 
                                sum(b.sale_price_usd) total_sales
                            FROM ORDERS_ICEBERG a 
                            INNER JOIN MENU b ON a.menu_item_id = b.menu_item_id
                            GROUP BY b.truck_brand_name
                            ORDER BY truck_brand_name
                        """)

    agg_df.write.mode("append").save_as_table("agg_sales_and_cost")
    return agg_df



