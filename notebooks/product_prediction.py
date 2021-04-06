#!/usr/bin/env python
# coding: utf-8

# In[48]:


get_ipython().magic(u'config Completer.use_jedi = False')

from pyspark.sql import SparkSession
import numpy
import pandas

import os
os.environ['PYSPARK_PYTHON'] = '/var/www/py_spark_ccf/PY_SPARK_CCF_ENV/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/var/www/py_spark_ccf/PY_SPARK_CCF_ENV/bin/python3'
os.getcwd()


# In[49]:


spark_session = SparkSession.builder.master("spark://costrategix-pc:7077")    .appName('product_prediction').getOrCreate()


# In[50]:


spark_session.sparkContext.getConf().getAll()


# In[51]:


audit_data_frame = spark_session.read.csv('../data/audit_data_frame_2021_04_02.csv',
    inferSchema=True, header=True)


# In[52]:


audit_data_frame.printSchema()


# In[53]:


for column in audit_data_frame.columns:
    audit_data_frame.select(column).describe().show()


# # feature extraction

# In[54]:


from tokenizer import tokenize
from pyspark.sql.functions import split
spark_tokenize = spark_session.udf.register('tokenizer', tokenize)
audit_data_frame = audit_data_frame.withColumn('INVOICE_PACKAGE_DESCRIPTION_CLEANED',
                                               split(spark_tokenize('INVOICE_PACKAGE_DESCRIPTION'), " "))
audit_data_frame.head(1)


# In[55]:


from pyspark.ml.feature import CountVectorizer, NGram, StringIndexer


# In[56]:


ngram_generator = NGram(n=2, inputCol='INVOICE_PACKAGE_DESCRIPTION_CLEANED',
                        outputCol='INVOICE_PACKAGE_DESCRIPTION_NGRAM')
audit_data_frame = ngram_generator.transform(audit_data_frame)
audit_data_frame.head(1)


# In[57]:


count_vec_1 = CountVectorizer(inputCol='INVOICE_PACKAGE_DESCRIPTION_CLEANED',outputCol='cnt_vec_1', minDF=4)
audit_data_frame = count_vec_1.fit(audit_data_frame).transform(audit_data_frame)
audit_data_frame.head(1)


# In[58]:


count_vec_2 = CountVectorizer(inputCol='INVOICE_PACKAGE_DESCRIPTION_NGRAM',outputCol='cnt_vec_2', minDF=4)
audit_data_frame = count_vec_2.fit(audit_data_frame).transform(audit_data_frame)
audit_data_frame.head(1)


# # add product_fdc_id

# In[59]:


entity_package_data_frame = pandas.read_csv('../data/catalog_with_price.csv')


# In[60]:


entity_package_data_frame.head()


# In[61]:


package_id_product_id_map = entity_package_data_frame.dropna(subset=['PACKAGE_FDC_ID', 'ESD_PRODUCT_FDC_ID'])    .set_index('PACKAGE_FDC_ID')['ESD_PRODUCT_FDC_ID'].to_dict()


# In[62]:


# package_id_product_id_map


# In[63]:


from pyspark.sql.types import NullType
audit_data_frame = audit_data_frame.dropna(subset=['PACKAGE_FDC_ID'])
get_product_id = spark_session.udf.register('get_product_id',
    lambda package_id: package_id_product_id_map[package_id] \
    if package_id in package_id_product_id_map else NullType())
audit_data_frame = audit_data_frame.withColumn('PRODUCT_FDC_ID', get_product_id('PACKAGE_FDC_ID'))
audit_data_frame = audit_data_frame.dropna(subset=['PRODUCT_FDC_ID'])
audit_data_frame.head(1)


# In[64]:


audit_data_frame.count()


# # data exploration

# In[65]:


audit_data_frame.createOrReplaceTempView("table1")
spark_session.sql("""
select PRODUCT_FDC_ID from table1
group by PRODUCT_FDC_ID having count(*) > 100;
""").count()


# In[66]:


product_row_list = spark_session.sql("""
select PRODUCT_FDC_ID from table1
group by PRODUCT_FDC_ID having count(*) > 100;
""").collect()

product_list = [row['PRODUCT_FDC_ID'] for row in product_row_list]
audit_data_frame = audit_data_frame.filter(audit_data_frame['PRODUCT_FDC_ID'].isin(product_list))
audit_data_frame.count()


# In[67]:


from pyspark.ml.feature import StringIndexer
str_indexer = StringIndexer(inputCol='PRODUCT_FDC_ID', outputCol='label')
audit_data_frame = str_indexer.fit(audit_data_frame).transform(audit_data_frame)
audit_data_frame.head(1)


# In[68]:


from pyspark.ml.feature import VectorAssembler
vec_assembler = VectorAssembler(inputCols=['cnt_vec_1', 'cnt_vec_2'], outputCol='features')
audit_data_frame = vec_assembler.transform(audit_data_frame)
audit_data_frame.head(1)


# # train test split

# In[69]:


final_data = audit_data_frame[['features', 'label']]
final_data.head(1)


# In[70]:


train_data, test_data = final_data.randomSplit([0.7, 0.3])


# # model training

# In[71]:


from pyspark.ml.classification import NaiveBayes


# In[72]:


model = NaiveBayes()
model = model.fit(train_data)


# # model evaluation

# In[73]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[74]:


acc_eval = MulticlassClassificationEvaluator()


# In[75]:


test_results = model.transform(test_data)


# In[76]:


test_results = test_results.filter(test_results['prediction'] > 0)


# In[77]:


test_results.count()


# In[78]:


print('F1')
acc_eval.evaluate(test_results)


# In[79]:


print('accuracy')
acc_eval.evaluate(test_results, {acc_eval.metricName: "accuracy"})


# In[ ]:




