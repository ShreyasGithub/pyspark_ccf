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


spark_session = SparkSession.builder.master("spark://costrategix-pc:7077")    .appName('simple_streaming').getOrCreate()


# In[3]:


spark_session.sparkContext.getConf().getAll()


# In[4]:


from pyspark.streaming import StreamingContext

batch_interval = 10
streaming_context = StreamingContext(spark_session.sparkContext, batch_interval)


# In[5]:


lines_dstream = streaming_context.socketTextStream("localhost", 9999)


# In[6]:


#shreyas encryptor
replace_dict = {
    'l': '1',
    's': '5',
    'e': '9',
    'a': '6',
}

encrypted_dstream = lines_dstream.map(    lambda line: ''.join([replace_dict[c] if c in replace_dict else c for c in line.lower()]))

encrypted_dstream.pprint()


# In[7]:


streaming_context.start()


# In[10]:


# streaming_context.stop()


# In[9]:


# nc -lk 9999

