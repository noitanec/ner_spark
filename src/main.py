import sparknlp
from sparknlp.pretrained import PretrainedPipeline

spark = sparknlp.start()

# Load pre-trained pipeline for NER
pipeline = PretrainedPipeline('recognize_entities_dl', lang='en')

# Sample DataFrame
data = spark.createDataFrame([["Google was founded in 1998 by Larry Page and Sergey Brin."]]).toDF("text")

# Run NER
result = pipeline.transform(data)
result.select("text", "entities").show(truncate=False)
