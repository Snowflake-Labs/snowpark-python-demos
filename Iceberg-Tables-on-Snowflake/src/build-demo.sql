-- create an external volume
CREATE OR REPLACE EXTERNAL VOLUME exvol
   STORAGE_LOCATIONS =
      (
         (
            NAME = 'my-s3-us-west-2'
            STORAGE_PROVIDER = 'S3'
            STORAGE_BASE_URL = 's3://build-demo-iceberg/iceberg-table/'
            STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::849350360261:role/sf-exvol-access'
            ENCRYPTION=(TYPE='AWS_SSE_KMS' KMS_KEY_ID='d0c718e2-fc88-4b7b-8db1-3ebaa5f85ef1')
         )
      );

DESC EXTERNAL VOLUME exvol;

-- create glue catalog integration
CREATE OR REPLACE CATALOG INTEGRATION sf_glue_catalog_integ
CATALOG_SOURCE=GLUE
CATALOG_NAMESPACE='build-iceberg-demo'
TABLE_FORMAT=ICEBERG
GLUE_AWS_ROLE_ARN='arn:aws:iam::849350360261:role/sf-glue-catalog-access'
GLUE_CATALOG_ID='849350360261'
GLUE_REGION='us-west-2'
ENABLED=TRUE;

DESCRIBE CATALOG INTEGRATION sf_glue_catalog_integ;

-- create iceberg table with glue as a catalog
CREATE OR REPLACE ICEBERG TABLE orders_iceberg
  EXTERNAL_VOLUME='exvol'
  CATALOG='sf_glue_catalog_integ'
  CATALOG_TABLE_NAME='orders_iceberg';

-- display the types of iceberg tables in the database
SHOW TABLES;

-- display row counts from the unmanaged iceberg table
SELECT COUNT(*) FROM orders_iceberg;

-- Convert the orders_iceberg to snowflake managed
ALTER ICEBERG TABLE orders_iceberg CONVERT TO MANAGED
    BASE_LOCATION='sf-catalog';

SHOW TABLES;

-- add some rows to the table
INSERT INTO orders_iceberg
    SELECT * 
    FROM orders_iceberg 
    LIMIT 5;

SELECT COUNT(*) FROM orders_iceberg;

-- Time travel on iceberg table
SELECT 
    COUNT(*) AS after_row_count
    , before_row_count
FROM orders_iceberg
JOIN (
        SELECT COUNT(*) AS before_row_count
        FROM orders_iceberg at(offset => -15)
    )
    ON 1=1
GROUP BY 2;

-- TODO: Show Snowpark transformations

-- display row counts from the managed iceberg table
SELECT COUNT(*) FROM agg_sales_and_cost;

SELECT * FROM agg_sales_and_cost LIMIT 5;

-- Governance: Row access policy based on brandname
CREATE ROLE bbq_only;

CREATE OR REPLACE ROW ACCESS POLICY vino_rap
AS (truck_brand_name VARCHAR(16777216)) RETURNS BOOLEAN -> 
    ('BBQ_ONLY'=CURRENT_ROLE() AND truck_brand_name='Smoky BBQ');

ALTER ICEBERG TABLE AGG_SALES_AND_COST ADD ROW ACCESS POLICY vino_rap ON (truck_brand_name);

GRANT ALL ON DATABASE BUILD_ICEBERG_DB TO ROLE BBQ_ONLY;
GRANT ALL ON SCHEMA BUILD_ICEBERG_DB.PUBLIC TO ROLE BBQ_ONLY;
GRANT ALL ON TABLE BUILD_ICEBERG_DB.PUBLIC.AGG_SALES_AND_COST TO ROLE BBQ_ONLY;
GRANT USAGE ON WAREHOUSE BUILD_ICEBERG_WH TO ROLE BBQ_ONLY;

-- read AGG_SALES_AND_COST table as BBQ_ONLY role
USE ROLE BBQ_ONLY;
USE WAREHOUSE BUILD_ICEBERG_WH;

SELECT * FROM AGG_SALES_AND_COST;

-- TODO: read snowflake managed table agg_sales_and_cost from Spark

-- end