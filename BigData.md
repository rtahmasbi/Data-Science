# Big Data
[Source](https://www.quora.com/What-are-some-software-and-skills-that-every-data-scientist-should-know-excluding-R-Matlab-and-Hadoop-Also-what-are-some-resources-for-learning-Hadoop)

- Hadoop: Apache Hadoop is an open-source software framework for storage and large-scale processing of data-sets on clusters of commodity hardware. Hadoop is an Apache top-level project being built and used by a global community of contributors and users.

- Spark: Apache Spark is an open-source data analytics cluster computing framework originally developed in the AMPLab at UC Berkeley. Spark fits into the Hadoop open-source community, building on top of the Hadoop Distributed File System (HDFS). However, Spark is not tied to the two-stage MapReduce paradigm, and promises performance up to 100 times faster than Hadoop MapReduce for certain applications.

- Bridges:


- SOLR:  Solr to build a highly scalable data analytics engine to enable customers to engage in lightning fast, real-time knowledge discovery.
Solr (pronounced "solar") is an open source enterprise search platform from the Apache Lucene project. Its major features include full-text search, hit highlighting, faceted search, dynamic clustering, database integration, and rich document (e.g., Word, PDF) handling. Providing distributed search and index replication, Solr is highly scalable. Solr is the most popular enterprise search engine. Solr 4 adds NoSQL features.

- S3: Amazon S3 is an online file storage web service offered by Amazon Web Services. Amazon S3 provides storage through web services interfaces.

- MapReduce: Hadoop MapReduce is a software framework for easily writing applications which process vast amounts of data (multi-terabyte data-sets) in-parallel on large clusters (thousands of nodes) of commodity hardware in a reliable, fault-tolerant manner.

- Corona: Corona, a new scheduling framework that separates cluster resource management from job coordination. Corona introduces a cluster managerwhose only purpose is to track the nodes in the cluster and the amount of free resources. A dedicated job tracker is created for each job, and can run either in the same process as the client (for small jobs) or as a separate process in the cluster (for large jobs).
One major difference from our previous Hadoop MapReduce implementation is that Corona uses push-based, rather than pull-based, scheduling. After the cluster manager receives resource requests from the job tracker, it pushes the resource grants back to the job tracker. Also, once the job tracker gets resource grants, it creates tasks and then pushes these tasks to the task trackers for running. There is no periodic heartbeat involved in this scheduling, so the scheduling latency is minimized.

- HBase: HBase is an open source, non-relational, distributed database modeled after Google's BigTable and written in Java. It is developed as part of Apache Software Foundation's Apache Hadoop project and runs on top of HDFS (Hadoop Distributed Filesystem), providing BigTable-like capabilities for Hadoop. That is, it provides a fault-tolerant way of storing large quantities ofsparse data (small amounts of information caught within a large collection of empty or unimportant data, such as finding the 50 largest items in a group of 2 billion records, or finding the non-zero items representing less than 0.1% of a huge collection).

- Zookeeper: Apache ZooKeeper is a software project of the Apache Software Foundation, providing an open source distributed configuration service, synchronization service, and naming registry for large distributed systems.[clarification needed] ZooKeeper was a sub project of Hadoop but is now a top-level project in its own right.


- Hive: Apache Hive is a data warehouse infrastructure built on top of Hadoop for providing data summarization, query, and analysis. While initially developed by Facebook, Apache Hive is now used and developed by other companies such asNetflix. Amazon maintains a software fork of Apache Hive that is included in Amazon Elastic MapReduce on Amazon Web Services.


- Mahout: Apache Mahout is a project of the Apache Software Foundation to produce free implementations of distributed or otherwisescalable machine learning algorithms focused primarily in the areas of collaborative filtering, clustering and classification. Many of the implementations use the Apache Hadoop platform. Mahout also provides Java libraries for common maths operations (focused on linear algebra and statistics) and primitive Java collections. Mahout is a work in progress; the number of implemented algorithms has grown quickly,[3] but various algorithms are still missing.

- Lucene: Lucene is a bunch of search-related and NLP tools but it's core feature is being a search index and retrieval system. It takes data from a store like HBase and indexes it for fast retrieval from a search query. Solr uses Lucene under the hood to provide a convenient REST API for indexing and searching data. ElasticSearch is similar to Solr.

- Sqoop: Sqoop is a command-line interface to back SQL data to a distributed warehouse. It's what you might use to snapshot and copy your database tables to a Hive warehouse every night.

- Hue: Hue is a web-based GUI to a subset of the above tools. Hue aggregates the most common Apache Hadoop components into a single interface and targets the user experience. Its main goal is to have the users "just use" Hadoop without worrying about the underlying complexity or using a command line

- Pregel and it's open source twin Giraph is a way to do graph algorithms on billions of nodes and trillions of edges over a cluster of machines. Notably, the MapReduce model is not well suited to graph processing so Hadoop/MapReduce are avoided in this model, but HDFS/GFS is still used as a data store.

- NLTK: The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for the Python programming language. NLTK includes graphical demonstrations and sample data. It is accompanied by a book that explains the underlying concepts behind the language processing tasks supported by the toolkit, plus a cookbook.
NLTK is intended to support research and teaching in NLP or closely related areas, including empirical linguistics, cognitive science, artificial intelligence, information retrieval, and machine learning.

- Freebase: Freebase is a large collaborative knowledge base consisting of metadata composed mainly by its community members. It is an online collection of structured data harvested from many sources, including individual 'wiki' contributions.

- DBPedia: DBpedia is a project aiming to extract structured content from the information created as part of the Wikipedia project. This structured information is then made available on the World Wide Web. DBpedia allows users to query relationships and properties associated with Wikipedia resources, including links to other related datasets.
