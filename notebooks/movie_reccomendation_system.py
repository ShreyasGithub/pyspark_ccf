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


ratings_data_frame.count()


# In[ ]:


ratings_data_frame.printSchema()


# In[ ]:


ratings_data_frame.show(vertical=True, n=5)


# In[ ]:


ratings_data_frame = ratings_data_frame.dropna()
ratings_data_frame.count()


# In[ ]:


ratings_data_frame.select('userId').distinct().count()


# In[ ]:


ratings_data_frame.select('movieId').distinct().count()


# In[ ]:


ratings_data_frame.createOrReplaceTempView("table1")
spark_session.sql("""
select movieId from table1
group by movieId having count(*) > 10000;
""").count()


# In[ ]:


movie_row_list = spark_session.sql("""
select movieId from table1
group by movieId having count(*) > 10000;
""").collect()

movie_list = [row['movieId'] for row in movie_row_list]


# In[ ]:


ratings_data_frame = ratings_data_frame.filter(ratings_data_frame['movieId'].isin(movie_list))


# In[ ]:


ratings_data_frame.count()


# In[ ]:


spark_session.sql("""
select userId from table1
group by userId having count(*) > 1000;
""").count()


# In[ ]:


user_row_list = spark_session.sql("""
select userId from table1
group by userId having count(*) > 1000;
""").collect()

user_list = [row['userId'] for row in user_row_list]


# In[ ]:


ratings_data_frame = ratings_data_frame.filter(ratings_data_frame['userId'].isin(user_list))


# In[ ]:


ratings_data_frame.count()


# In[ ]:


train_data, test_data = ratings_data_frame.randomSplit([0.7, 0.3])


# In[ ]:


from pyspark.ml.recommendation import ALS
model = ALS(maxIter=10, userCol="userId", itemCol="movieId", ratingCol="rating")


# In[ ]:


model = model.fit(train_data)


# In[ ]:


test_data.head(1)


# In[ ]:


test_user_data = test_data.filter(test_data['userId'] == 229)


# In[ ]:


test_user_data.collect()


# In[ ]:


single_user = test_user_data.select(['movieId','userId'])


# In[ ]:


reccomendations = model.transform(single_user)
reccomendations.orderBy('movieId').collect()


# In[ ]:


from pyspark.ml.evaluation import RegressionEvaluator


# In[ ]:


test_data.count()


# In[ ]:


test_results = model.transform(test_data)


# In[ ]:


test_results.head(5)


# In[ ]:


evaluator = RegressionEvaluator(labelCol='rating', predictionCol='prediction')
print('RMSE')
evaluator.evaluate(test_results)


# In[ ]:


print('R_sqr')
evaluator.evaluate(test_results, {evaluator.metricName: "r2"})


# In[ ]:


print('MAE')
evaluator.evaluate(test_results, {evaluator.metricName: "mae"})


# In[ ]:


test_data.select('rating').describe().show()


# In[ ]:




