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


spark_session = SparkSession.builder.master("spark://costrategix-pc:7077")    .appName('product_prediction').getOrCreate()


# In[ ]:


spark_session.sparkContext.getConf().getAll()


# In[ ]:


audit_data_frame = spark_session.read.csv('../data/audit_data_frame.csv', inferSchema=True, header=True)


# In[ ]:


audit_data_frame.printSchema()


# In[ ]:


for column in audit_data_frame.columns:
    audit_data_frame.select(column).describe().show()


# # feature extraction

# In[ ]:


from tokenizer import tokenize
from pyspark.sql.functions import split
spark_tokenize = spark_session.udf.register('tokenizer', tokenize)
audit_data_frame = audit_data_frame.withColumn('INVOICE_PACKAGE_DESCRIPTION_CLEANED',
                                               split(spark_tokenize('INVOICE_PACKAGE_DESCRIPTION'), " "))
audit_data_frame.head(1)


# In[ ]:


from pyspark.ml.feature import CountVectorizer, NGram, StringIndexer


# In[ ]:


ngram_generator = NGram(n=2, inputCol='INVOICE_PACKAGE_DESCRIPTION_CLEANED',
                        outputCol='INVOICE_PACKAGE_DESCRIPTION_NGRAM')
audit_data_frame = ngram_generator.transform(audit_data_frame)
audit_data_frame.head(1)


# In[ ]:


count_vec_1 = CountVectorizer(inputCol='INVOICE_PACKAGE_DESCRIPTION_CLEANED',outputCol='cnt_vec_1', minDF=4)
audit_data_frame = count_vec_1.fit(audit_data_frame).transform(audit_data_frame)
audit_data_frame.head(1)


# In[ ]:


count_vec_2 = CountVectorizer(inputCol='INVOICE_PACKAGE_DESCRIPTION_NGRAM',outputCol='cnt_vec_2', minDF=4)
audit_data_frame = count_vec_2.fit(audit_data_frame).transform(audit_data_frame)
audit_data_frame.head(1)


# # add product_fdc_id

# In[ ]:


entity_package_data_frame = pandas.read_csv('../data/catalog_with_price.csv')


# In[ ]:


entity_package_data_frame.head()


# In[ ]:


package_id_product_id_map = entity_package_data_frame.dropna(subset=['PACKAGE_FDC_ID', 'ESD_PRODUCT_FDC_ID'])    .set_index('PACKAGE_FDC_ID')['ESD_PRODUCT_FDC_ID'].to_dict()


# In[ ]:


# package_id_product_id_map


# In[ ]:


from pyspark.sql.types import NullType
audit_data_frame = audit_data_frame.dropna(subset=['PACKAGE_FDC_ID'])
get_product_id = spark_session.udf.register('get_product_id',
    lambda package_id: package_id_product_id_map[package_id] \
    if package_id in package_id_product_id_map else NullType())
audit_data_frame = audit_data_frame.withColumn('PRODUCT_FDC_ID', get_product_id('PACKAGE_FDC_ID'))
audit_data_frame = audit_data_frame.dropna(subset=['PRODUCT_FDC_ID'])
audit_data_frame.head(1)


# In[ ]:


audit_data_frame.count()


# In[ ]:


from pyspark.ml.feature import StringIndexer
str_indexer = StringIndexer(inputCol='PRODUCT_FDC_ID', outputCol='label')
audit_data_frame = str_indexer.fit(audit_data_frame).transform(audit_data_frame)
audit_data_frame.head(1)


# In[ ]:


from pyspark.ml.feature import VectorAssembler
vec_assembler = VectorAssembler(inputCols=['cnt_vec_1', 'cnt_vec_2'], outputCol='features')
audit_data_frame = vec_assembler.transform(audit_data_frame)
audit_data_frame.head(1)


# # train test split

# In[ ]:


final_data = audit_data_frame[['features', 'label']]
final_data.head(1)


# In[ ]:


train_data, test_data = final_data.randomSplit([0.7, 0.3])


# # model training

# In[ ]:


from pyspark.ml.classification import NaiveBayes


# In[ ]:


model = NaiveBayes()
model = model.fit(train_data)


# # model evaluation

# In[ ]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[ ]:


acc_eval = MulticlassClassificationEvaluator()


# In[ ]:


test_results = model.transform(test_data)


# In[ ]:


test_results = test_results.filter(test_results['prediction'] > 0)


# In[ ]:


test_results.count()


# In[ ]:


print('F1')
acc_eval.evaluate(test_results)


# In[ ]:


print('accuracy')
acc_eval.evaluate(test_results, {acc_eval.metricName: "accuracy"})


# In[ ]:




