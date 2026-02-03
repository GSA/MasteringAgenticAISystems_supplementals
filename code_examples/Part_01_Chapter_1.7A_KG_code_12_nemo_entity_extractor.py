from nemo.collections.nlp.models import TokenClassificationModel

class NeMoEntityExtractor:
    """GPU-accelerated entity extraction using NVIDIA NeMo."""

    def __init__(self):
        # Load NVIDIA pre-trained NER model
        self.ner_model = TokenClassificationModel.from_pretrained("ner_en_bert")

    def extract_entities(self, text: str) -> Set[Tuple[str, str]]:
        """
        Extract entities using NVIDIA NeMo NER with GPU acceleration.

        Args:
            text: Input text

        Returns:
            Set of (entity_text, entity_type) tuples
        """
        predictions = self.ner_model.predict([text])

        entities = set()
        for prediction in predictions[0]:
            entity_text = prediction['entity']
            entity_type = prediction['label']
            entities.add((entity_text, entity_type))

        return entities
