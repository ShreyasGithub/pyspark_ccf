#!/usr/bin/env python
# coding: utf-8

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


spark_session = SparkSession.builder.master("spark://costrategix-pc:7077")    .appName('movie_reccomendation_system').getOrCreate()


# In[ ]:


spark_session.sparkContext.getConf().getAll()


# In[ ]:


ratings_data_frame = spark_session.read.csv('../data/ratings.csv', inferSchema=True, header=True)


# In[ ]:


ratings_data_frame.printSchema()


# In[ ]:


ratings_data_frame.show(vertical=True, n=5)


# In[ ]:


train_data, test_data = ratings_data_frame.randomSplit([0.7, 0.3])


# In[ ]:


from pyspark.ml.recommendation import ALS
model = ALS(maxIter=4, userCol="userId", itemCol="movieId", ratingCol="rating")


# In[ ]:


model = model.fit(train_data)


# In[ ]:


test_data.head(1)


# In[ ]:


test_user_data = test_data.filter(test_data['userId'] == 1)


# In[ ]:


test_user_data.collect()


# In[ ]:


single_user = test_user_data.select(['movieId','userId'])


# In[ ]:


reccomendations = model.transform(single_user)
reccomendations.collect()


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


test_user_data.select('ratings').describe().show()

