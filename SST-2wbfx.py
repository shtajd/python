import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, rand
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# 设置Java路径
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"


# 创建Spark会话
def create_spark_session():
    return SparkSession.builder \
        .appName("SST2-Text-Classification") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.master", "local[*]") \
        .getOrCreate()


# 从本地读取SST-2数据集
def load_local_sst2_data(data_path):
    """从本地路径加载SST-2数据集"""
    train_path = os.path.join(data_path, "train.tsv")
    dev_path = os.path.join(data_path, "dev.tsv")

    # 读取训练集和开发集
    train_df = pd.read_csv(train_path, sep='\t')
    dev_df = pd.read_csv(dev_path, sep='\t')

    return train_df, dev_df


# 将Pandas DataFrame转换为Spark DataFrame
def convert_to_spark_dataframe(spark, train_df, dev_df):
    # 转换为Spark DataFrame
    train_spark_df = spark.createDataFrame(train_df)
    dev_spark_df = spark.createDataFrame(dev_df)

    # 重命名列
    train_spark_df = train_spark_df.withColumnRenamed("sentence", "text").withColumnRenamed("label", "label")
    dev_spark_df = dev_spark_df.withColumnRenamed("sentence", "text").withColumnRenamed("label", "label")

    return train_spark_df, dev_spark_df


# 传统机器学习方法：使用Spark ML Pipeline + 参数调优
def traditional_machine_learning_pipeline(train_df, dev_df):
    print("开始传统机器学习方法...")

    # 定义基础Pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashing_tf = HashingTF(inputCol="filtered_words", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    # 评估器
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")

    # ------------------
    # 逻辑回归模型调参
    # ------------------
    print("开始逻辑回归模型调参...")
    lr = LogisticRegression(labelCol="label", featuresCol="features")

    # 定义参数网格
    lr_param_grid = ParamGridBuilder() \
        .addGrid(hashing_tf.numFeatures, [5000, 10000, 20000]) \
        .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    # 创建交叉验证器
    lr_crossval = CrossValidator(
        estimator=Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, lr]),
        estimatorParamMaps=lr_param_grid,
        evaluator=evaluator_accuracy,
        numFolds=3,
        seed=42
    )

    # 运行交叉验证
    lr_cv_model = lr_crossval.fit(train_df)

    # 获取最优模型
    best_lr_model = lr_cv_model.bestModel

    # 评估最优模型
    lr_predictions = best_lr_model.transform(dev_df)
    lr_accuracy = evaluator_accuracy.evaluate(lr_predictions)

    # 打印最优参数
    best_lr_params = {
        "numFeatures": best_lr_model.stages[2].getNumFeatures(),
        "regParam": best_lr_model.stages[4].getRegParam(),
        "elasticNetParam": best_lr_model.stages[4].getElasticNetParam()
    }

    print(f"逻辑回归最优参数: {best_lr_params}")
    print(f"逻辑回归模型准确率: {lr_accuracy:.4f}")

    # ------------------
    # 随机森林模型调参
    # ------------------
    print("\n开始随机森林模型调参...")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")

    # 定义参数网格
    rf_param_grid = ParamGridBuilder() \
        .addGrid(hashing_tf.numFeatures, [5000, 10000]) \
        .addGrid(rf.numTrees, [50, 100, 200]) \
        .addGrid(rf.maxDepth, [5, 10, 20]) \
        .build()

    # 创建交叉验证器
    rf_crossval = CrossValidator(
        estimator=Pipeline(stages=[tokenizer, stopwords_remover, hashing_tf, idf, rf]),
        estimatorParamMaps=rf_param_grid,
        evaluator=evaluator_accuracy,
        numFolds=3,
        seed=42
    )

    # 运行交叉验证
    rf_cv_model = rf_crossval.fit(train_df)

    # 获取最优模型
    best_rf_model = rf_cv_model.bestModel

    # 评估最优模型
    rf_predictions = best_rf_model.transform(dev_df)
    rf_accuracy = evaluator_accuracy.evaluate(rf_predictions)

    # 打印最优参数
    best_rf_params = {
        "numFeatures": best_rf_model.stages[2].getNumFeatures(),
        "numTrees": best_rf_model.stages[4].getNumTrees,
        "maxDepth": best_rf_model.stages[4].getMaxDepth()
    }

    print(f"随机森林最优参数: {best_rf_params}")
    print(f"随机森林模型准确率: {rf_accuracy:.4f}")

    return best_lr_model, lr_predictions, best_rf_model, rf_predictions


def main():
    data_path = "/home/sql/PycharmProjects/pythonProject/SST-2"

    # 创建Spark会话
    spark = create_spark_session()

    # 加载数据
    train_df, dev_df = load_local_sst2_data(data_path)

    # 转换为Spark DataFrame
    train_spark_df, dev_spark_df = convert_to_spark_dataframe(spark, train_df, dev_df)

    # 运行传统机器学习方法（带调参）
    traditional_machine_learning_pipeline(train_spark_df, dev_spark_df)

    # 保持Spark会话打开，直到用户手动关闭
    input("按Enter键退出...")
    spark.stop()


if __name__ == "__main__":
    main()