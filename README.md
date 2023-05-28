# AWS Machine Learning Specialty Cheatsheet

## Table Of Contents
[Day 1](#day-1)  
[Day 2](#day-2)  
[Day 3](#day-3)  
[Day 4](#day-4)  
[Day 5](#day-5)  
[Day 6](#day-6)

---

## Day 1
`Apache Spark` -> An open-source unified analytics engine for **large-scale** data processing.  
`Object2Vec` -> A general-purpose neural **embedding** algorithm that is highly customizable. It can learn **low-dimensional** dense embeddings of *high-dimensional* objects.    
`Semantic segmentation` -> Supervised, Categorize each **pixel** in an image into a class or object.   
`Amazon SageMaker Object Detection` -> MXNet algorithm detects and **classifies** objects in images using a single deep neural network.   
`Amazon Rekognition` -> Offers pre-trained and customizable computer vision (CV) capabilities to **extract** information(*metadata*) and insights from your images and videos. **Celebrity** detection(Note: can't handle the very specific classification task)   
`Random Cut Forest` -> unsupervised, detect **anomalous** data points within a data set.    
`Elasticsearch(Amazon)` -> A distributed, RESTful *search* and *analytics* engine capable of addressing a growing number of use cases. Individual **server** required.   
`SageMaker Ground Truth` -> A data **labeling service** that makes it easy to label data(option: *Amazon Mechanical Turk*)   
`Inference pipeline` -> Preprocessing, predictions, and post-processing on real-time and batch inference requests.  
`Amazon IoT Greengrass` -> Software, extends *cloud capabilities* to **local devices**.  
`Amazon SageMaker Neo` -> Enables developers to **optimize** machine learning (ML) models for inference on SageMaker in the cloud and supported devices at the **edge**.   
`Nvidia jetson edge` -> AI computing platform for GPU-accelerated parallel processing in mobile embedded ... Robotics and Edge Computing.  
`AWS DeepLens` -> *Hardware*, a deep learning-enabled **video camera**.  
`Amazon EMR` -> A managed cluster platform that simplifies running big data frameworks, such as **Apache Hadoop** and **Apache Spark**.(File system: *hdfs*, emrfs, local file system)   
`Splunk` -> Search, analyze and visualize all of data   
`K-nearest neighbors(KNN)` -> Supervised, classification, regression. Uses proximity to make classifications or predictions about the **grouping** of an individual data point.  
`Sparkmagic` -> A set of **tools** for interactively working with remote *Spark clusters* through Livy, a Spark REST server, in Jupyter notebooks.  
`XGBoost`  -> An *Extreme Gradient Boosting algorithm* that is optimized for boosted **decision trees**.  
`SageMakerEstimator` -> Tight integration between *Spark* and *SageMaker* for several models including XGBoost, and offers the **simplest** solution  
`MLLib` -> Built on top of *Spark*, MLlib is a scalable **machine learning library** consisting of common learning algorithms and utilities, including classification, regression, clustering, collaborative filtering, dimensionality reduction, and underlying optimization primitives.  

### Kinesis
| Name | Note |
| --- | --- |
| Kinesis Data Stream | Streaming(**real-time**) ingest shards: 1MB/s or 1000 messages/s |
| Kinesis Data Analytics | **Near real-time** analytics(SQL query), scripts out the **unneeded** data. |
| Kinesis Data Firehose | **Not real-time**, load streams(**not** used to *stream video*) into S3, etc. **Ingest**, *JSON* -> *Parquet* or *ORC* |
| Kinesis Video Stream | streaming video(uses Amazon S3 for backend storage) |

### Date Imputation(Missing Data)
| Method | Type |
| --- | --- |
| KNN | Numerical |
| Deep Learning | Categorical | 
| MICE(Multiple Imputation by Chained Equations) | Finds relationships between features |
| Create a separate boolean column | - |
| Fill with zeros | - |
| Regression | - |
|Impute with median value | Outlier exists |

### Merics
1. Accuracy
$$Accuracy=\frac{TP+TN}{TP+TN+FP+FN}$$
* **Overall** performance

2. Precision
$$precision=\frac{TP}{TP+FP}$$
* Minimize **false positives**

3. Recall
$$recall=\frac{TP}{TP+FN}$$
* Minimize **false negatives**, describes positive records that predicted correctly.

4. F1 Score
$$F1=2*\frac{precision*recall}{precision+recall}$$
* **Balancing** between Precision and Recall

Where `TP/FP` - `True/False Positive`, `FP/FN` - `False Positive/Negative`, respectively.  

[Top](#aws-machine-learning-specialty-cheatsheet)

---

## Day 2
`Qualitative data` -> Non-numeric forms of data.  \
`T-SNE(T-Distributed Stochastic Neighbor Embedding)` -> An unsupervised, non-linear technique primarily used for data exploration, **dimensionality reduction** and **visualizing high-dimensional** data.  
`Box plot` -> A method for graphically demonstrating the locality, spread and **skewness** groups of numerical data.  
`Histogram` -> A graph used to represent the frequency distribution of a few data points of one variable.  
`Scatterplot` -> Uses dots to represent values for **two different** numeric variables.  
`One-hot encoding` -> A process by which **categorical** variables are converted into a form that could be provided to ML algorithms.  
`AWS Batch` -> AWS Batch helps you to run batch computing workloads on the AWS Cloud. **Automate** the batch *data preprocessing* and ML training aspects of the *pipeline*. **Scheduling** and **allocating** the resources    
`AWS Glue` -> A **serverless** Apache **Spark** platform, *data preprocessing* for analysis through automated extract, transform and load (**ETL**) processes. **S3 prefixes**  
`Principal component analysis(PCA)` -> A learning algorithm that **reduces** the **dimensionality** (number of features) within a dataset while retaining as much information as possible.  
`Seaborn distribution plot(distplot)` -> Depicts the variation in the **data distribution**.  
`Amazon Personalize` -> A fully managed machine learning service that uses your data to generate item **recommendations** for your users.  
`Amazon Textract` -> A machine learning (ML) service that automatically **extracts** text, handwriting, and data from scanned documents.  
`SMOTE(Synthetic Minority Oversampling Technique)` -> A technique to **up-sample** the minority classes while avoiding overfitting.  
`K-fold cross validation` -> A procedure used to estimate the **skill of the model** on new data.  
`One-class SVM` -> unsupervised, learns a decision function for **novelty** detection.  
`Seq2Seq` -> nlp, takes a sequence of items (words, letters, time series, etc) and outputs another sequence of items.  
`Bayesian optimization` -> Build a probability model of the objective function and uses it to select **hyperparameter** to evaluate in the true objective function.   
`Breadth-first search(BFS)` -> An algorithm for searching a tree *data structure* for a node that satisfies a given property.  
`Random Search(RS)` -> A technique where random combinations of the **hyperparameters** are used to find the best solution for the built model.  
`Grid Search` -> A tuning technique that attempts to compute the optimum values of **hyperparameters**.  
`Depth-first search` -> An algorithm for searching a graph or tree *data structure*.  
`Amazon FSx for Lustre` -> FSx for Lustre **speeds up** your training jobs by serving your Amazon S3 data to Amazon SageMaker at high speeds.  
`Amazon EBS(Amazon Elastic Block Store)` -> provides block level storage volumes for use with EC2 instances.  
`Amazon Lex` -> Enables you to build applications using a speech or text interface powered by the same technology that powers **Amazon Alexa**.  
`Amazon Polly` -> Use deep learning technologies to synthesize natural-sounding **human speech**. *Speech Marks*: starts and ends, *SSML*: control, *Lexicons*: customize  
`Amazon Transcribe` -> Converts **audio input into text**, which opens the door for various text analytics applications on voice input.  
`Amazon Quicksight` -> **Anomaly detection**, **forecasting**, **auto-narrative**: customize(personalized dashboard)  
`Amazon Athena` -> An interactive query service that makes it easy to analyze data directly in Amazon Simple Storage Service (**Amazon S3**) using standard **SQL**. **Serverless**   
`Canary Deployment` -> In the beginning, the current version receives **100%** of user traffic  
`Xavier Initialization` -> An attempt to improve the initialization of neural network **weighted inputs**, in order to avoid some traditional problems in machine learning.  

### Scaler
| Name | Explanation |
| --- | --- |
| MinMaxScaler | No-Gaussian distribution |
| StandardScaler | Gaussian distribution,  the features need to be standardized to ensure that they contribute **equally** to the analysis.

### Optimizer
| Name | Explanation |
| --- | --- |
| Mini-batch gradient descent | Split the dataset into small subsets (batches) and compute the gradients for each batch. |
| Gradient descent | An optimization algorithm that's used when training a machine learning model. |
| **Adagrad** | An algorithm for gradient-based optimization. | 
| **Rmsprop** | An extension of gradient descent and the AdaGrad version of gradient descent that uses a decaying average of partial gradients in the adaptation of the step size for each parameter. |

### L1, L2 Regularation
| Name | Description |
| --- | --- |
| L1 | Reduce **dimensionality** |
| L2 | Adjust **weight** for features |

### Note
1. SageMaker, to update an `Endpoint`, you must create a new `EndpointConfig`
2. SageMaker, with `pipe input mode`, dataset is **streamed directly** to training instance, instead of being *downloaded* first.
3. **Parquet** is faster then *csv*. 
4. To use SageMaker training, the data should be split into training, validation, and test sets.
5. Correlation: Feature1(0.64) >(**stronger**) Feature2(-0.85) 
6. SageMaker, "Data download failed." -> Check IAM role of encrypt and decrypt the data
7. SageMaker, Hyperparameter tunning job: `HyperparameterTunner()`

[Top](#aws-machine-learning-specialty-cheatsheet)

---

## Day 3
`Word2Vec` -> A text **classification** algorithm. Word2vec is useful for sentiment analysis, entity recognition, and translation.  
`ROC(Receiver Operating Characteristic Curve)` -> A graph showing the performance of a classification model at all classification thresholds. A good ROC: (0, 1)   
`AUC(Area Under the ROC)` -> AUC measures the ability of the model to predict a higher score for positive examples as compared to negative examples.  
`K-means` -> Unsupervised, A **cluster** refers to a collection of data points aggregated together because of certain *similarities*.  
`Logistic regression` -> Supervised, find the relationships between two data factors(**binary output**).  
`LDA(Latent Dirichlet Allocation)` -> Unsupervised, **classification**, a **topic** modelling technique that can classify text in a *document* to a particular topic. Same as `NTM`.  
`Amazon SageMaker Autopilot` -> **Automatically** trains and tunes the best machine learning models for classification or regression, based on your data while allowing to maintain full control and visibility.  
`Apache Flink` -> A **streaming** dataflow engine that you can use to run real-time stream processing on high-throughput data sources.  
`Amazon SageMaker BlazingText` -> Provides highly optimized implementations of the **Word2vec(relationships)** and **text classification** algorithms.  
`Amazon SageMaker NTM(Neural Topic Model)` -> Unsupervised, used to organize a corpus of **documents** into topics that contain word **groupings** based on their statistical distribution. Same as `LDA`.  
`Peered VPCs` -> A networking connection between two VPCs that enables you to route traffic between them using private IPv4 addresses or IPv6 addresses. (Data does **not traverse** the **public** internet.)  
`A/B testing` -> **Compares** the performance of two versions of content to see which one *appeals* more to visitors/viewer.  
`Blue/Green deployment` -> An application release model that **gradually transfers** user traffic from a previous version of an app or microservice to a nearly identical new release—both of which are running in production.  
`Amazon Macie` -> A **data security service** that uses machine learning (ML) and pattern matching to discover and help protect your sensitive data.  
`FM(Factorization Machines)` -> Supervised, a general-purpose supervised learning algorithm that you can use for both **classification** and **regression** tasks. **float32** format.  
`Amazon SageMaker Linear Learner` -> Supervised, used for solving either **classification** or **regression** problems.  
`AWS Panorama` -> Add **computer vision(CV)** to your existing fleet of cameras with AWS Panorama devices, which integrate seamlessly with your **local** area network.  
`AWS DeepRacer` -> An autonomous 1/18th scale race car designed to test **RL** models by racing on a physical track.  
`Amazon Augmented AI` -> Implement **human reviews** and audits of ML predictions based on your specific requirements, including multiple reviewers.

### EC2 Types
| Option | Discount |
| --- | ---|
| On-Demand | 0% |
| Reserved | 40%-60% |
| Spot | 50%-90% |

### Note
1. SageMaker `endpoints` are not targets for `Application Load Balancers`.
2. `AWS Glue` cannot run **SQL** queries on the data.
3. `Amazon Lex` top choice, `Slot Value`

[Top](#aws-machine-learning-specialty-cheatsheet)

---

## Day 4 
`Kinesis Producer Library (KPL)` -> Simplifies producer application development, allowing developers to achieve high write **throughput** to a *Kinesis Data Stream*.   
`Kinesis Client Library(KCL)` -> Acts as an intermediary between your record **processing** logic and *Kinesis Data Streams*.  
`Data Augmentation` -> A set of techniques to artificially **increase** the amount of data by generating **new data** points from existing data.  
`Amazon SageMaker DeepAR` -> A supervised learning algorithm for forecasting scalar (one-dimensional) time series using *Recurrent Neural Networks* (**RNN**).  
`Image localization` -> Aims to locate **the main single** (or most visible) object in an image.  
`Image Classification` -> Assigning a **label** or class to an entire image.  
`Instance Segmentation` -> Deals with detecting instances of objects and demarcating their **boundaries**.  
`Weight Decay` -> A **regularization** technique by adding a small penalty, usually the L2 norm of the weights (all the weights of the model).  
`SageMaker Elastic Inference` -> Allows you to attach a **low-cost GPU** to your instance.  
`Anaconda` -> An open-source distribution of the Python and R programming languages for **data science** that aims to simplify package management and deployment.  
`DescribeTrainingJob(SageMaker)` -> Returns information about a job you previously initiated, check **FailureReason**.  
`AWS CloudWatch` -> A **monitoring** service for AWS resources and applications.  
`AWS CloudTrail` -> A web service that records **API** activity in your AWS account.  
`Amazon RDS` -> Amazon Relational Database Service (Amazon RDS) is a web service that makes it easier to set up, operate, and scale a **relational database** in the AWS Cloud.  
`Amazon Redshift Spectrum` -> Data warehousing service that lets a data analyst conduct fast, complex analysis on objects stored on the AWS cloud.
Provides *Online Analytical Processing* (**OLAP**).  
`Pair plot` -> Plot **pairwise relationships** in a dataset.  
`Box and Whisker plot` -> Shows how the data is **distributed** and it also shows any **outliers**.  
`Tree map` -> An alternative way of visualising the hierarchical structure of a **Tree Diagram** while also displaying **quantities** for each category via area size.  
`Multilayer Perceptron (MLP)` -> A **fully connected** multi-layer neural network.  
`Autoregressive Integrated Moving Average (ARIMA)` -> A method for forecasting or predicting *future* outcomes based on a historical **time series**.  
`Amazon Forecast` -> A fully managed service that uses statistical and machine learning algorithms to deliver highly accurate **time series** forecasts.  
`AutoML` -> Choose the *best model* for your time-series data. (including **DeepAR**, **ARIMA**, etc)  
`Sockeye` -> A sequence-to-sequence framework for **Neural Machine Translation** based on Apache MXNet Incubating.  
`Service control policies (SCPs)` -> A type of **organization policy** that you can use to manage permissions in your organization.  
`Organizational Unit (OU)` -> A logical **grouping of accounts** in your organization, created using AWS Organizations.

### S3
| Name | Type | Note |
| --- | --- | --- |
| Amazon S3 Standard | **Frequent** |Processed data |
| Amazon S3 Glacier Instant Retrieval | **Rarely** accessed, requires retrieval in milliseconds. | Processed data |
| Amazon S3 Glacier Deep Archive | Accessed **once** or twice in a year | Raw data |
* `S3 Lifecycle Rule`: Automating the archiving or deletion of old data

### Data Distribution
| Name | Explanation |
| --- | --- |
| Normal distribution | Data near the **mean** are more frequent in occurrence than data far from the mean.|
| Poisson distribution| Expresses the probability of a given number of events occurring in a **fixed** interval of time or space if these events occur with a known constant mean rate and independently of the time since the last event. |
| Binomial distribution | The discrete probability distribution that gives only **two possible results** in an experiment, either success or failure. |
| Bernoulli distribution | **Multiple trials**(compare binomial distribution) |

### Note
1. `DynamoDB Stream` can only be used in `DynamoDB`.
2. `SageMaker Estimator local mode` can quickly experiment with the models without having to wait for the training data to be **loaded**.
3. A single `Amazon SageMaker endpoint` cannot serve two different `models`.
4. Fraud Detection(*positive: fraud*), false **negative**: incorrectly identifying fraudulent transactions as **non-fraudulent**.

[Top](#aws-machine-learning-specialty-cheatsheet)

---
## Day 5
`Within-cluster sum of squares(WSS)` -> Determining the **optimal value** of **k** in *k-Means* clustering.  
`Custom inference container` -> *Amazon SageMaker*, port **8080**, requests under **2s**, compress in **tar** format.  
`Incremental learning` -> A machine learning method where **new data** is incrementally added to a model, and the model is retrained on the new data.  
`ResNet-50` -> A convolutional neural network(**CNN**) that is 50 layers deep.  
`Boosting` -> A method used in machine learning to **reduce errors** in predictive data analysis.  
`Binning(Interval Binning)` -> The process of transforming *numerical* variables into their **categorical** counterparts.  
`Quantile Binning` -> The process of assigning the **same** number of observations to each **bin** if the number of observations is evenly divisible by the number of bins.  
`Horovod` -> A **distributed deep learning** training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. **Multiple GPU**  

### Note
1. `Hyperparameter tunning`: Run only **one** training job at a time.

[Top](#aws-machine-learning-specialty-cheatsheet)

---
## Day 6
`Support Vector Machine(SVM) with RBF kernel` -> Supervised, for **classification** and **regression**, can also be used for **dimensionality reduction**.  
`Amazon Aurora` -> A fully managed relational database engine that's compatible with *MySQL* and *PostgreSQL*.  
`AWS Glue's FindMatches` -> A new way to perform **de-duplication** as part of *Glue ETL*.  
`RecordIO` -> The best **format** choice on models.  
`Amazon SQS` -> Send, store, and receive messages between software components at any volume.  
`AWS Step Functions` -> A visual **workflow service** that helps developers use AWS services to build distributed applications, automate processes, orchestrate microservices, and create data and machine learning (ML) pipelines.  
`Amazon Simple Workflow Service` -> A fully managed **workflow service** for building scalable, resilient applications. More complex then *AWS Step Functions*.  
`EMR File System (EMRFS)` -> An implementation of HDFS that all Amazon EMR clusters use for **reading and writing** regular files from Amazon EMR directly to *Amazon S3*.  
`LibSVM` -> A specific **data format** for the data analysis tool LibSVM. *Glue ETL*, *Kinesis Analytics* are not supported.  
`ImageNet` -> A large visual **database** designed for use in visual object recognition software research.  

### Note
1. A Spot interruption on the **Master node** will *terminate* the entire *cluster*, and on a **Core node**, it can lead to HDFS *data loss*.
2. **Vanishing gradient** results from small derivates of the **sigmoid** activation function, try **ReLU**.  
3. `BlazingText`: Labels must be prefixed with `__label__`, and the tokens within the sentence - including punctuation - should be `space` separated.
4. `BlazingText Word2Vec mode`: The order of words **doesn't** matter, skip-gram and continuous bag of words(**CBOW**) architectures. Find *relationships* between **individual words**.
5. **XGBoost 1.0/1.1**: CPU-only algorithms, **XGBoost 1.2/newer**: GPU, Accelerated Computing(EC2, `P3`)
6. **Bring Your Own Containers**: Set `ENV SAGEMAKER_PROGRAM` for `train.py` in the Dockerfile.
7. `SageMaker inter-container traffic encryption`, secured in-transit of private VPC.
8. A large *learning rate* will **overshoot** the true minima, while a small will **slow down** convergence.
9. A large *batch size* will **get stuck** in localized minima.
10. SageMaker **default IAM role**, buckets' name with "sagemaker" is accessible.

[Top](#aws-machine-learning-specialty-cheatsheet)

---
## Day 7
`AWS PrivateLink` -> Establish connectivity between **VPCs** and AWS services without exposing data to the internet.  
`Beta Testing` -> Acceptance Testing, the end users evaluate its performance.  

### Note
1. **SSE-KMS** is used for the Amazon **S3** service. 
2. `Kinesis Data Firehose` + `built-in lambda` is eaiser than `Kinesis Data Streams` + `Glue ETL`, as a `Glue` script is required.(for *near real-time*)

[Top](#aws-machine-learning-specialty-cheatsheet)

---
## Resources
* AWS Skill Builder -> AWS Certified Machine Learning - Specialty Official Practice Question Set (MLS-C01 - English)
* AWS Skill Builder -> Exam Readiness: AWS Certified Machine Learning - Specialty
* Udemy -> AWS Certified Machine Learning Specialty 2023 - Hands On!
* Udemy -> AWS Certified Machine Learning Specialty Full Practice Exam
* Udemy -> AWS Certified Machine Learning Specialty: 3 PRACTICE EXAMS

---


