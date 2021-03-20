#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'config Completer.use_jedi = False')


# In[ ]:


from pyspark.sql import SparkSession
import numpy
import pandas


# In[ ]:


import os
os.environ['PYSPARK_PYTHON'] = '/var/www/py_spark_ccf/PY_SPARK_CCF_ENV/bin/python3'
os.getcwd()


# In[ ]:


spark_session = SparkSession.builder.master("spark://costrategix-pc:7077")    .appName('explore_data_frame').getOrCreate()


# In[ ]:


spark_session.sparkContext.getConf().getAll()


# In[ ]:


food_data_frame = spark_session.read.csv('../data/food.csv', inferSchema=True, header=True)


# In[ ]:


food_data_frame.printSchema()


# In[ ]:


food_data_frame.show()


# In[ ]:


food_data_frame.groupBy('data_type').count().show()


# In[ ]:


food_data_frame.select('description').show()


# In[ ]:


str_len = spark_session.udf.register('str_len', lambda text: len(text))
description_len_data_frame = food_data_frame.select(str_len('description'))
description_len_data_frame.show()


# In[ ]:


description_len_data_frame['str_len(description)']


# In[ ]:


food_data_frame = food_data_frame.withColumn('description_len',str_len('description'))
food_data_frame.show()


# In[ ]:


food_data_frame.head()


# In[ ]:


def distributed_fuzzy_ratio(text_1, text_2='Metabolizable Energy of Almonds'):
    from fuzzywuzzy import fuzz
    return fuzz.ratio(text_1, text_2)
    
fuzzy_match = spark_session.udf.register('fuzzy_match', distributed_fuzzy_ratio)


# In[ ]:


food_data_frame = food_data_frame.withColumn('match_score',     fuzzy_match('description'))
food_data_frame.show()


# In[ ]:


food_data_frame.filter((food_data_frame['description'].like('%Metab%'))).show()


# In[ ]:


food_data_frame.filter(food_data_frame['data_type'] == 'experimental_food').show()


# In[ ]:




