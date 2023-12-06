use role accountadmin;

create or replace database build_iceberg_db;
create or replace warehouse build_iceberg_wh;

use database build_iceberg_db;
use warehouse build_iceberg_wh;

create or replace TABLE MENU (
	MENU_ID NUMBER(19,0),
	MENU_TYPE_ID NUMBER(38,0),
	MENU_TYPE VARCHAR(16777216),
	TRUCK_BRAND_NAME VARCHAR(16777216),
	MENU_ITEM_ID NUMBER(38,0),
	MENU_ITEM_NAME VARCHAR(16777216),
	ITEM_CATEGORY VARCHAR(16777216),
	ITEM_SUBCATEGORY VARCHAR(16777216),
	COST_OF_GOODS_USD NUMBER(38,4),
	SALE_PRICE_USD NUMBER(38,4)
);

-- TODO: load the data into the standard snowflake table manually

create or replace iceberg table agg_sales_and_cost (
    TRUCK_BRAND_NAME VARCHAR(16777216),
    TOTAL_COST NUMBER(38,4),
    TOTAL_PRICE NUMBER(38,4)
)
    catalog = 'SNOWFLAKE'
    external_volume = 'exvol'
    base_location = 'agg-sales-cost';


alter user vinod set rsa_public_key="MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA783Gz7ppyiTJyP+WvU5rEG76b7B5k90ev9f3EALsIDMmedSuwZeqcxdhE31tgu+SwjiCjhDAK+y7SlPWVxIXEnVTSrGBmIJL8EKUC1QtylIx9g1VGwEPTh/PB5qmvgQXliOe2X8FO214D4FAPcIJzk0AQHJ1U9X67tcIqBmXhcKD1FAhBNPxPANNlFoVDIhxI1LJCodCwSwCLJCcpvMr0y3/7mwFclVrvRk1hY8QpSzMdPZqYvjeQ/GIx/BVdQla2GsYT0+R75/FDCBIvzMywYoqmuDYorHhdX3r9mfyWW9GlJmOkRIci9SsBYRX+B+8seRx6yu4xgnZs04CW0Kq5wIDAQAB";



