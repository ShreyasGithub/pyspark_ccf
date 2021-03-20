#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Problem Statement
A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.

They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:

Which variables are significant in predicting the price of a car
How well those variables describe the price of a car
Based on various market surveys, the consulting firm has gathered a large data set of different types of cars across the America market."""


# In[ ]:


get_ipython().magic(u'config Completer.use_jedi = False')

from pyspark.sql import SparkSession
import numpy
import pandas

import os
os.environ['PYSPARK_PYTHON'] = '/var/www/py_spark_ccf/PY_SPARK_CCF_ENV/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/var/www/py_spark_ccf/PY_SPARK_CCF_ENV/bin/python3'
os.getcwd()


# In[ ]:


spark_session = SparkSession.builder.master("spark://costrategix-pc:7077")    .appName('car_price_prediction').getOrCreate()


# In[ ]:


spark_session.sparkContext.getConf().getAll()


# In[ ]:


car_data_frame = spark_session.read.csv('../data/CarPrice_Assignment.csv', inferSchema=True, header=True)


# In[ ]:


car_data_frame.printSchema()


# In[ ]:


car_data_frame.show(vertical=True, n=5)


# In[ ]:


columnList = [item[0] for item in car_data_frame.dtypes if not item[1].startswith('string')]
for column in columnList:
    car_data_frame.select(column).describe().show()


# # data profiling

# In[ ]:


from pyspark.sql.functions import isnull, when, count, col
nacounts = car_data_frame.select([count(when(isnull(c), c)).alias(c) for c in car_data_frame.columns])
nacounts.show(vertical=True)


# # feature extraction

# In[ ]:


from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder


# In[ ]:


car_data_frame.columns


# In[ ]:


categorical_columns = ['fueltype',
 'aspiration',
 'doornumber',
 'carbody',
 'drivewheel',
 'enginelocation',
'enginetype',
 'cylindernumber',
'fuelsystem']


# In[ ]:


string_index_encoder = StringIndexer(inputCols=categorical_columns,
                                outputCols=[c + '_str_ind' for c in categorical_columns],
                                    stringOrderType='alphabetAsc')
train_car_data_frame = string_index_encoder.fit(car_data_frame).transform(car_data_frame)
train_car_data_frame.head(1)


# In[ ]:


one_hot_encoder = OneHotEncoder(inputCols=[c + '_str_ind' for c in categorical_columns],
                                outputCols=[c + '_vec' for c in categorical_columns],
                               dropLast=False)
train_car_data_frame = one_hot_encoder.fit(train_car_data_frame).transform(train_car_data_frame)
train_car_data_frame.head(1)


# In[ ]:


numeric_columns = ['wheelbase',
                   'carlength',
 'carwidth',
 'carheight',
 'curbweight',
'enginesize',
 'boreratio',
 'stroke',
 'compressionratio',
 'horsepower',
 'peakrpm',
 'citympg',
 'highwaympg',]


# In[ ]:


from pyspark.ml.feature import VectorAssembler

vector_assembler = VectorAssembler(inputCols=[c + '_vec' for c in categorical_columns] + numeric_columns,
                                  outputCol='unscaled_features')
vector_data = vector_assembler.transform(train_car_data_frame)


# In[ ]:


vector_data.head(1)


# In[ ]:


vector_data = vector_data.withColumnRenamed('price', 'label')
final_data = vector_data[['unscaled_features', 'label']]
final_data.head(1)


# In[ ]:


from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol='unscaled_features', outputCol='features')
final_data = scaler.fit(final_data).transform(final_data)
    
final_data.head(1)


# # split train/test

# In[ ]:


train_data, test_data = final_data.randomSplit([0.7,0.3])


# # Model training

# In[ ]:


from pyspark.ml.regression import RandomForestRegressor
model = RandomForestRegressor(numTrees=100)
model = model.fit(train_data)


# # model evaluation

# In[ ]:


model.featureImportances


# In[ ]:


from pyspark.ml.evaluation import RegressionEvaluator


# In[ ]:


test_results = model.transform(test_data)


# In[ ]:


evaluator = RegressionEvaluator()
print('RMSE')
evaluator.evaluate(test_results)


# In[ ]:


print('R_sqr')
evaluator.evaluate(test_results, {evaluator.metricName: "r2"})


# In[ ]:


print('MAE')
evaluator.evaluate(test_results, {evaluator.metricName: "mae"})


# In[ ]:


car_data_frame.select('price').describe().show()


# In[ ]:


transformed_column_names = []
for column in [c for c in categorical_columns]:
    for row in car_data_frame.select(column).distinct().orderBy(column).collect():
#         print(column, row[column])
        transformed_column_names.append(column + '_' + row[column])

transformed_column_names = transformed_column_names + numeric_columns
transformed_column_names


# In[ ]:


len(transformed_column_names)


# In[ ]:


for ind, importance in enumerate(model.featureImportances):
    print(transformed_column_names[ind], round(importance, 2))


# In[ ]:


from pyspark.sql.functions import desc


# In[ ]:




