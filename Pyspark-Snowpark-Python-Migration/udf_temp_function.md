```python
# "PySpark Code Sample"
<!--- PySpark Code Sample -->
@udf(returnType=StringType())
def toLower(val:str):
    if val is not None:
        return val.lower()
      
df_udf.select(col("EmployeeName"),toLower(col("EmployeeName"))\
              .alias("LowerCaseEmployeeName")).show()
```

```python
<!--- Snowpark Python Code Sample -->
@udf(return_type==StringType(),replace=True)
def toLower(val:str):
    if val is not None:
        return val.lower()
      
df_udf.select(col("EmployeeName"),toLower(col("EmployeeName"))\
              .alias("LowerCaseEmployeeName")).show()
```