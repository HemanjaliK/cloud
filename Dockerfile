FROM openjdk:8-jre-slim

# Set build arguments for labels and Spark version
ARG BUILD_DATE
ARG SPARK_VERSION=3.0.0

# Set Docker labels
LABEL org.label-schema.name="Apache PySpark $SPARK_VERSION" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.version=$SPARK_VERSION

# Set environment variables
ENV PATH="/opt/miniconda3/bin:${PATH}"
ENV PYSPARK_PYTHON="/opt/miniconda3/bin/python"
ENV PROG_DIR /winepredict

# Install required dependencies
RUN set -ex && \
    apt-get update && \
    apt-get install -y curl bzip2 --no-install-recommends && \
    curl -s -L --url "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" --output /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -f -p "/opt/miniconda3" && \
    rm /tmp/miniconda.sh && \
    conda config --set auto_update_conda true && \
    conda config --set channel_priority false && \
    conda update conda -y --force-reinstall && \
    conda clean -tipy && \
    echo "PATH=/opt/miniconda3/bin:${PATH}" > /etc/profile.d/miniconda.sh && \
    pip install --no-cache pyspark==${SPARK_VERSION} numpy pandas scikit-learn && \
    SPARK_HOME=$(python /opt/miniconda3/bin/find_spark_home.py) && \
    echo "export SPARK_HOME=$(python /opt/miniconda3/bin/find_spark_home.py)" > /etc/profile.d/spark.sh && \
    curl -s -L --url "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.7.4/aws-java-sdk-1.7.4.jar" --output $SPARK_HOME/jars/aws-java-sdk-1.7.4.jar && \
    curl -s -L --url "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/2.7.3/hadoop-aws-2.7.3.jar" --output $SPARK_HOME/jars/hadoop-aws-2.7.3.jar && \
    mkdir -p $SPARK_HOME/conf && \
    echo "spark.hadoop.fs.s3.impl=org.apache.hadoop.fs.s3a.S3AFileSystem" >> $SPARK_HOME/conf/spark-defaults.conf && \
    apt-get remove -y curl bzip2 && \
    apt-get autoremove -y && \
    apt-get clean

# Copy application files
COPY test.py /winepredict/
COPY ValidationDataset.csv /winepredict/
COPY trainingmodel.model /winepredict/

# Set the working directory
WORKDIR $PROG_DIR

# Set the entry point and default command
ENTRYPOINT ["spark-submit"]
CMD ["test.py", "ValidationDataset.csv"]
