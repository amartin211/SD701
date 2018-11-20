# Databricks notebook source
# MAGIC %md #Kaggle Competition
# MAGIC 
# MAGIC In this session lab, we are going to compete in a Kaggle Competition.
# MAGIC 
# MAGIC First, we are going to upload the `train` and `test` datasets to databricks using the following route:
# MAGIC 
# MAGIC *Data -> Add Data -> Upload File*
# MAGIC 
# MAGIC **Note:** You have the option to select the location to store the files within DBFS.

# COMMAND ----------

# MAGIC %md Once the files are uploaded, we can use them in our environment.
# MAGIC 
# MAGIC You will need to change /FileStore/tables/train.csv with the name of the files and the path(s) that you chose to store them.
# MAGIC 
# MAGIC **Note 1:** When the upload is complete, you will get a confirmation along the path and name assigned. Filenames might be slightly modified by Databricks.
# MAGIC 
# MAGIC **Note 2:** If you missed the path and filename message you can navigate the DBFS via: *Data -> Add Data -> Upload File -> DBFS* or checking the content of the path `display(dbutils.fs.ls("dbfs:/FileStore/some_path"))`

# COMMAND ----------

train_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferSchema='true').load('/FileStore/tables/train_set-51e11.csv')

test_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferSchema='true').load('/FileStore/tables/test_set-b5f57.csv')

display(train_data)

# COMMAND ----------

print('Train data size: {} rows, {} columns'.format(train_data.count(), len(train_data.columns)))
print('Test data size: {} rows, {} columns'.format(test_data.count(), len(test_data.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC We will use the `VectorAssembler()` to merge our feature columns into a single vector column as requiered by Spark methods.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vector_assembler1 = VectorAssembler(inputCols=["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34", "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40"], outputCol="features")




# COMMAND ----------

#Intuitively, we can see that some features are continuous (from 'elevation' to 'Horizontal_Distance_To_Fire_Points') while other are categorical (all features from Wilderness_Area1 to Soil_Type40). 
# Also there aren't any null values and the data appears to be clean.

# COMMAND ----------

# MAGIC %md For this example, we will use `Logistic Regression`.

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import PCA
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import Binarizer
from pyspark.ml.feature import Normalizer
from pyspark.ml import Pipeline
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# COMMAND ----------

# Features enginering:

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
featureIndexer = VectorIndexer(inputCol="scaledFeatures", outputCol="indexFeatures", maxCategories=2)
pca = PCA(k=35, inputCol="scaledFeatures", outputCol="pcaFeatures") # Keeping 35 features 

#Interestingly, none of the above features enginering processes have improved my accuracy. In fact, adding these processes into my pipeline has reduced my accuracy score regardless of the ML algorithm chosen. 

# COMMAND ----------

#ML algorithms used:
# I've chosed this set of algorithms as I thought they would do relatively well for a multi classification problem.
# My approach has consisted in quickly trying few algorithms to see what was the accuracy. After making some research online about parameters optimization for multiclass problems I modified heuristically my parameters so as to maximize my accuracy. 

dt = DecisionTreeClassifier(labelCol="Cover_Type", featuresCol="pcaFeatures", maxDepth=30, maxBins=85,impurity='entropy') # BEST ONE with this set of parameters #(Accuracy is about 90% on the actual test set.)

rf = RandomForestRegressor(labelCol="Cover_Type", featuresCol="pcaFeatures", numTrees=200,maxDepth=30)

classifier = LogisticRegression(labelCol="Cover_Type", featuresCol="scaledFeatures",maxIter=10, regParam=0.3, elasticNetParam=0.8)

layers = [54,10,9,8]
trainer = MultilayerPerceptronClassifier(maxIter=300, layers=layers, blockSize=128,labelCol="Cover_Type", featuresCol="scaledFeatures",seed=1234)
  

# COMMAND ----------

# Spliting the data into a training and test sets:

(training, test) = train_data.randomSplit([0.8, 0.2])

# COMMAND ----------

# Defining the Pipeline

pipeline = Pipeline(stages=[vector_assembler1,scaler,pca,dt]) 


# COMMAND ----------

# Defining a ParamGrid to search for best parameters 
paramGrid = ParamGridBuilder().addGrid(dt.maxDepth, [5,10,15,30]).addGrid(dt.maxBins, [10,40,70,100,130]).build()


# COMMAND ----------

# Running the model with the paramGrid defined above

dtm = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator=MulticlassClassificationEvaluator(metricName="f1", labelCol="Cover_Type",predictionCol="prediction"),
                           trainRatio=0.8)
model_dtm = dtm.fit(training)



#Prediction of the model on the test set
evaluator=MulticlassClassificationEvaluator(metricName="f1", labelCol="Cover_Type",predictionCol="prediction")
new_pred = model_dtm.transform(test)
accuracy = evaluator.evaluate(new_pred)
print("Accuracy = %g " % (accuracy))

# COMMAND ----------

# This section refer to another approach.
# I've tried to run first an unsupervised model to categorized the data into 7 catogeries (labels). I removed the column cover_type prior to running a Kmeans model with K=7. 

from pyspark.ml.clustering import KMeans
kmeans = KMeans().setK(7).setSeed(1) # setting Kmeans with K 7 category. 



# Training data: integrating the kmeans prediction into my features 

pipeline1 = Pipeline(stages=[vector_assembler1,kmeans])
model_kmeans = pipeline.fit(training) 
new_trainingdata = model_kmeans.transform(training)
vector_assembler2 = VectorAssembler(inputCols=["features","prediction"], outputCol="feature")
new_train = vector_assembler2.transform(new_trainingdata)


#Test Data => doing likewise for the test data
model_kmeans = pipeline.fit(test)
new_testdata = model_kmeans.transform(test)
vector_assembler2 = VectorAssembler(inputCols=["features","prediction"], outputCol="feature")
new_test = vector_assembler2.transform(new_testdata)
redtest_data = new_test.select(col("Id"),col("feature")) # used to compute accuracy


# The next step consisted of using the predictions from the kmeans model and use it as an additional feature which would then be fed into my decision tree model.

# fitting my new training set to a decision tree model 
model_DTkmeans = dt.fit(new_train)


# prediction for the test set
new_pred = dt.transform(redtest_data)

# evaluation of the prediction
evaluator=MulticlassClassificationEvaluator(metricName="f1", labelCol="Cover_Type",predictionCol="prediction")
accuracy = evaluator.evaluate(new_pred)
print("Accuracy = %g " % (accuracy))

# This model did not performed as expected as I had a slightly lower score on the actual test set.

# COMMAND ----------

# Finally, in order to compare my previous results I wanted to try another algorithm which I thought would do particularly well, namely the Gboost classifier via the sklearn library. The code below reflect my last performance score on Kaggle.

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path1=('/Users/alexmartin/Downloads/all-3/train-set.csv')
path2=('/Users/alexmartin/Downloads/all-3/test-set.csv') 
train_data = pd.read_csv(path1)
test_data = pd.read_csv(path2)

X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


model = XGBClassifier(max_depth=8,learning_rate=0.1, n_estimators=1000, verbose=True,colsample_bytree=0.8)
model.fit(X_data, y_train)

y_pred = model.predict(test_data)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))



# COMMAND ----------

#This part is the one from the initial TP

# Make predictions on testData
predictions = model_dtm.transform(test_data)


predictions = predictions.withColumn("Cover_Type", predictions["prediction"].cast("int"))  # Cast predictions to 'int' to mach the data type expected by Kaggle
# Show the content of 'predictions'
predictions.printSchema()

#accuracy = evaluator.evaluate(predictions)
predictions.count()

# COMMAND ----------

# Display predictions and probabilities
display(predictions.select("Cover_Type", "probability"))


# COMMAND ----------

# MAGIC %md Finally, we can create a file with the predictions.

# COMMAND ----------

# Select columns Id and prediction
(predictions
 .repartition(1)
 .select('Id', 'Cover_Type')
 .write
 .format('com.databricks.spark.csv')
 .options(header='true')
 .mode('overwrite')
 .save('/FileStore/kaggle-submission'))
 

# COMMAND ----------

# MAGIC %md To be able to download the predictions file, we need its name (`part-*.csv`):

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/FileStore/kaggle-submission"))


# COMMAND ----------

# MAGIC %md Files stored in /FileStore are accessible in your web browser via `https://<databricks-instance-name>.cloud.databricks.com/files/`.
# MAGIC   
# MAGIC For this example:
# MAGIC 
# MAGIC https://community.cloud.databricks.com/files/kaggle-submission/part-*.csv?o=######
# MAGIC 
# MAGIC where `part-*.csv` should be replaced by the name displayed in your system  and the number after `o=` is the same as in your Community Edition URL.
# MAGIC 
# MAGIC 
# MAGIC Finally, we can upload the predictions to kaggle and check what is the perfromance.
