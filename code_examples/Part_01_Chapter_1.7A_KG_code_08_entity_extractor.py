import spacy
from typing import Set, Tuple

class EntityExtractor:
    """Extracts named entities from text using spaCy NER."""

    def __init__(self, model: str = "en_core_web_lg"):
        """
        Initialize entity extractor with spaCy model.

        Args:
            model: spaCy model name (en_core_web_lg recommended for accuracy)
        """
        self.nlp = spacy.load(model)

    def extract_entities(self, text: str) -> Set[Tuple[str, str]]:
        """
        Extract named entities from text.

        Args:
            text: Input document text

        Returns:
            Set of (entity_text, entity_type) tuples

        Example:
            >>> extractor.extract_entities("Elon Musk founded SpaceX in 2002")
            {("Elon Musk", "PERSON"), ("SpaceX", "ORG"), ("2002", "DATE")}
        """
        doc = self.nlp(text)
        entities = set()

        for ent in doc.ents:
            # Filter to relevant entity types for business knowledge graphs
            if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "DATE", "MONEY"]:
                # Normalize whitespace and casing
                normalized_text = " ".join(ent.text.split())
                entities.add((normalized_text, ent.label_))

        return entities
