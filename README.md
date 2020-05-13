# H2O3-NiFi-MiNiFi-Integration
Integrate H2O3 MOJO into NiFi &amp; MiNiFi C++

Set your environment variables for EC2 instance pem key and public DNS. If on mac os x, set them in `~/.bash_profile` else if on linux, set them in `~/.profile`.

- `H2O3_SCORING_PEM={pem-key}`
- `H2O3_SCORING_INSTANCE={public-dns}`

~~~bash
ssh -i $H2O3_SCORING_PEM ubuntu@$H2O3_SCORING_INSTANCE
~~~

Clone GitHub Repo:

~~~bash
git clone https://github.com/james94/H2O3-NiFi-MiNiFi-Integration
~~~

The following commands were used to create directory structure, I include them in case you may want to recreate the directory
structure in another project:

~~~bash
# Create directory structure for H2O-3 model deployment projects

# Create directory for prepData
mkdir -p model-deployment/common/hydraulic/prepData/

# Create directory for testData
mkdir -p model-deployment/common/{license,hydraulic/testData/{test-batch-data,test-real-time-data}}

# Create directory for predData
mkdir -p model-deployment/common/hydraulic/predData/{pred-real-time-data,pred-batch-data}

# Create directory for mojo scoring pipeline
mkdir -p model-deployment/common/hydraulic/mojo-model

# mkdir -p model-deployment/common/hydraulic/{mojo-scoring-pipeline/{java-runtime/java/mojo2-java-runtime,cpp-runtime/{c/mojo2-c-runtime/{linux,macos-x,ibm-powerpc},r-wrapper/mojo2-r-runtime/{linux,macos-x,ibm-powerpc},python-wrapper/mojo2-py-runtime/{linux,macos-x,ibm-powerpc}}},python-scoring-pipeline}

# Create directory for NiFi and MiNiFi C++
mkdir -p model-deployment/apps/{java,nifi/{nifi-nar-bundles,templates},nifi-minifi-cpp/{conf,minifi-python/h2o/h2o3/msp}}
tree model-deployment
~~~

Install H2O3 using Conda

~~~bash
conda create -n h2o3-nifi-minifi python=3.6
conda activate h2o3-nifi-minifi
conda config --append channels conda-forge
conda install -y -c h2oai h2o
pip install datatable
pip install pandas
# Make all packages available on EC2 instance
sudo apt-get -y update
# Install Java to include open source H2O-3 algorithms
sudo apt-get -y install openjdk-8-jdk
~~~

Practice AutoML

/Users/jmedel/Development/James/H2O3-NiFi-MiNiFi-Integration/model-deployment/common/hydraulic/prepData

~~~python
# Build Model with AutoML
import os
import h2o
import matplotlib as plt
from h2o.automl import H2OAutoML

h2o.init()

hydraulic_data = h2o.import_file(os.path.join(os.environ['HOME'], "Development", "James", "H2O3-NiFi-MiNiFi-Integration", "model-deployment", "common", "hydraulic", "prepData", "hydraulicData_coolCondY.csv"))

# print first 10 rows
hydraulic_data.head()

# print statistical summary
hydraulic_data.describe()

# look at cool_cond_y, which is the response of our classification problem
hydraulic_data["cool_cond_y"].table()

# visualize cool_cond_y in hist for our classification problem
hydraulic_data["cool_cond_y"].hist()

train, test = hydraulic_data.split_frame([0.75], seed=42)

# print the split
print("train:%d test:%d" % (train.nrows, test.nrows))

# select training column fieldnames and exclude the label fieldname
x = train.names
y = "cool_cond_y" # fieldname of the label we are using to train the model
x.remove(y)

# For multinomial classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML (limited to 900 seconds max runtime, 42 seeds for reproducibility)
aml = H2OAutoML(max_runtime_secs=900, seed=42, project_name='classifyHydCoolCond')
# optional: x is set to all columns being the predictors other than the label
# train the model on predicting cool_cond_y using all columns
aml.train(x=x, y=y, training_frame=train)

# print the leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)

# The leader model is stored here
aml.leader

# see how the best model performs on the test set
aml.leader.model_performance(test_data=test)

# let's make some predictions on our test set
preds = aml.predict(test)

# make some predictions using the leader model
preds = aml.leader.predict(test)
~~~


Saving Test Batch Data

~~~python
save_testData_path = os.path.join(os.environ['HOME'], "Development", "James", "H2O3-NiFi-MiNiFi-Integration", "model-deployment", "common", "hydraulic", "testData", "test-batch-data", "test_large_batch.csv")

# save batch data first 15 rows (small batch)
# grab range of rows from 0 to 4, stop at 5, but don't include 5.
test_first_15rows = test[range(0,15,1),:]

# save batch data all rows 551 (large batch)
# h2o.export_file(test, save_testData_path)

h2o.export_file(test_first_15rows, save_testData_path)
~~~

Saving Test Real-Time Data

~~~python
save_testData_path = os.path.join(os.environ['HOME'], "Development", "James", "H2O3-NiFi-MiNiFi-Integration", "model-deployment", "common", "hydraulic", "testData", "test-real-time-data")

# save batch data first 15 rows
# grab row at index 0
# test_row_at0 = test[0,:]

for i in range(0, 15, 1):
    h2o.export_file(test[i,:], os.path.join(save_testData_path, "test_" + str(i) + ".csv"))
~~~



Practice Saving, Loading, Downloading and Uploading Models

/Users/jmedel/Development/James/H2O3-NiFi-MiNiFi-Integration/model-deployment/common/hydraulic/mojo-pipeline

~~~python
path = os.path.join(os.environ['HOME'], "Development", "James", "H2O3-NiFi-MiNiFi-Integration", "model-deployment", "common", "hydraulic", "mojo-model")

# Get a list of model ids from auto ml leaderboard
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])

# Get the first model_ids
model_ids[0]

# Get first model from AutoML leaderboard
best_model = h2o.get_model(model_ids[0])

# need to save the model I want to keep individually as a MOJO
best_model.save_mojo(path)
~~~





Practice Productionizing H2O Models

Predict on Real-Time Data using H2O Frame

~~~python
import os
import h2o
h2o.init()

path = os.path.join(os.environ['HOME'], "Development", "James", "H2O3-NiFi-MiNiFi-Integration", "model-deployment", "common", "hydraulic", "mojo-model", "GBM_grid__1_AutoML_20200511_075150_model_180.zip")

imported_model = h2o.import_mojo(path)

testData_path = os.path.join(os.environ['HOME'], "Development", "James", "H2O3-NiFi-MiNiFi-Integration", "model-deployment", "common", "hydraulic", "testData", "test-real-time-data", "test_0.csv")

test_h2o_frame = h2o.import_file(testData_path)

# column headers is an issue, the H2O3 MOJO does not have a feature_names pred label fieldnames like DAI MOJO
first_line_header = False
input_schema = []

# does test dt frame column names (header) equal m_scorer feature_names (exp_header)
if first_line_header == False:
    test_h2o_frame.names = 

# test is an h2o frame, so instead of using datatable, I guess I will use h2o frame
predictions = imported_model.predict(test_h2o_frame)
~~~

Predict on Real-Time Data using H2O Frame, but first convert from datatable to numpy to H2O Frame

~~~python
import os
import h2o
import datatable as dt
h2o.init()

path = os.path.join(os.environ['HOME'], "Development", "James", "H2O3-NiFi-MiNiFi-Integration", "model-deployment", "common", "hydraulic", "mojo-model", "GBM_grid__1_AutoML_20200511_075150_model_180.zip")

imported_model = h2o.import_mojo(path)

testData_path = os.path.join(os.environ['HOME'], "Development", "James", "H2O3-NiFi-MiNiFi-Integration", "model-deployment", "common", "hydraulic", "testData", "test-real-time-data", "test_0.csv")

# Load csv data into Datatable
test_dt_frame = dt.fread(testData_path)

# convert datatable frame to h2oframe
# load tabular data str of 1 or more rows into datatable frame
test_h2o_frame = h2o.H2OFrame(python_obj=test_dt_frame.to_numpy(), column_names=list(test_dt_frame.names))

predictions = imported_model.predict(test_h2o_frame)

# Optional: Update the prediction names
# cool_cond_y,cool_cond_y.3,cool_cond_y.20,cool_cond_y.100
predictions.names = ['cool_cond_y','cool_cond_y.3','cool_cond_y.20','cool_cond_y.100']
~~~

Predict on Batch Data

~~~python
import h2o
h2o.init()

imported_model = h2o.import_mojo(path)

testData_path = os.path.join(os.environ['HOME'], "Development", "James", "H2O3-NiFi-MiNiFi-Integration", "model-deployment", "common", "hydraulic", "testData", "test-batch-data", "test.csv")

test_h2o_frame = h2o.import_file(testData_path)

predictions = imported_model.predict(test_h2o_frame)
~~~