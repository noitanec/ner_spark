import sparknlp
from sparknlp.pretrained import PretrainedPipeline

spark = sparknlp.start()


class NER:
    def __init__(self) -> None:
        self.pipeline = PretrainedPipeline('recognize_entities_dl', lang='en')

    def predict(self, data):
        return self.pipeline.transform(data)