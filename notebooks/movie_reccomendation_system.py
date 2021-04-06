#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic(u'config Completer.use_jedi = False')

from pyspark.sql import SparkSession
import numpy
import pandas

import os
os.environ['PYSPARK_PYTHON'] = '/var/www/py_spark_ccf/PY_SPARK_CCF_ENV/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/var/www/py_spark_ccf/PY_SPARK_CCF_ENV/bin/python3'
os.getcwd()


# In[2]:


spark_session = SparkSession.builder.master("spark://costrategix-pc:7077")    .appName('movie_reccomendation_system').getOrCreate()


# In[3]:


spark_session.sparkContext.getConf().getAll()


# In[4]:


ratings_data_frame = spark_session.read.csv('../data/ratings.csv', inferSchema=True, header=True)


# In[5]:


ratings_data_frame.count()


# In[6]:


ratings_data_frame.printSchema()


# In[7]:


ratings_data_frame.show(vertical=True, n=5)


# In[8]:


ratings_data_frame = ratings_data_frame.dropna()
ratings_data_frame.count()


# In[9]:


ratings_data_frame.select('userId').distinct().count()


# In[10]:


ratings_data_frame.select('movieId').distinct().count()


# In[11]:


ratings_data_frame.createOrReplaceTempView("table1")
spark_session.sql("""
select movieId from table1
group by movieId having count(*) > 10000;
""").count()


# In[12]:


movie_row_list = spark_session.sql("""
select movieId from table1
group by movieId having count(*) > 10000;
""").collect()

movie_list = [row['movieId'] for row in movie_row_list]


# In[13]:


ratings_data_frame = ratings_data_frame.filter(ratings_data_frame['movieId'].isin(movie_list))


# In[14]:


ratings_data_frame.count()


# In[15]:


spark_session.sql("""
select userId from table1
group by userId having count(*) > 1000;
""").count()


# In[16]:


user_row_list = spark_session.sql("""
select userId from table1
group by userId having count(*) > 1000;
""").collect()

user_list = [row['userId'] for row in user_row_list]


# In[17]:


ratings_data_frame = ratings_data_frame.filter(ratings_data_frame['userId'].isin(user_list))


# In[18]:


ratings_data_frame.count()


# In[19]:


train_data, test_data = ratings_data_frame.randomSplit([0.7, 0.3])


# In[20]:


from pyspark.ml.recommendation import ALS
model = ALS(maxIter=10, userCol="userId", itemCol="movieId", ratingCol="rating")


# In[21]:


model = model.fit(train_data)


# In[22]:


test_data.head(1)


# In[23]:


test_user_data = test_data.filter(test_data['userId'] == 229)


# In[24]:


test_user_data.collect()


# In[25]:


single_user = test_user_data.select(['movieId','userId'])


# In[26]:


reccomendations = model.transform(single_user)
reccomendations.orderBy('movieId').collect()


# In[27]:


from pyspark.ml.evaluation import RegressionEvaluator


# In[28]:


test_data.count()


# In[29]:


test_results = model.transform(test_data)


# In[30]:


test_results.head(5)


# In[31]:


evaluator = RegressionEvaluator(labelCol='rating', predictionCol='prediction')
print('RMSE')
evaluator.evaluate(test_results)


# In[32]:


print('R_sqr')
evaluator.evaluate(test_results, {evaluator.metricName: "r2"})


# In[33]:


print('MAE')
evaluator.evaluate(test_results, {evaluator.metricName: "mae"})


# In[34]:


test_data.select('rating').describe().show()


# In[ ]:




