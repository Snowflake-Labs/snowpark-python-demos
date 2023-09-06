-- The purpose of this test is to showcase how the data pipeline will work. We use existing data
-- to fill the APPLICATION_RECORD table to showcase how it triggers downstreams processes.
-- In real-life, APPLICATION_RECORD table will have data feeding from other sources.

-- Run this code to run batch inference process.
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE SSK_RESEARCH;
USE DATABASE ML_SNOWPARK_CI_CD;


-- Step 1: Insert new data into the stream. Lets add 2 rows.
INSERT INTO ML_SNOWPARK_CI_CD.DATA_PROCESSING.APPLICATION_RECORD
(SELECT * FROM ML_SNOWPARK_CI_CD.DATA_PROCESSING.APPLICATION_RECORD LIMIT 2);

-- Step 2: Output the stream table - you should get 2 rows.
SELECT * from DATA_PROCESSING.APPLICATION_RECORD_STREAM;

-- Step 3: Execute the TASK - it should run on the 2 rows. If the stream is empty, it will get skipped.
-- Step 3 will trigger model inference.
EXECUTE TASK ML_PROCESSING.TASK_PROCESS_INPUT;

-- Wait for Step 3 to execute before executing Step 4: Output results from the Scored_data table to see if the function worked.
SELECT * FROM ML_PROCESSING.SCORED_DATA ORDER BY PREDICTION_TIMESTAMP DESC;