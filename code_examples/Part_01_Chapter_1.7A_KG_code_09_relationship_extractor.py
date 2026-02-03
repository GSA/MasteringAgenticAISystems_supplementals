from typing import List

class RelationshipExtractor:
    """Extracts relationships between entities using dependency parsing."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

        # Map verbs to standardized relationship types
        self.verb_mapping = {
            "found": "FOUNDED",
            "establish": "FOUNDED",
            "create": "CREATED",
            "lead": "LEADS",
            "manage": "MANAGES",
            "own": "OWNS",
            "invest": "INVESTED_IN",
            "acquire": "ACQUIRED",
            "partner": "PARTNERED_WITH",
            "compete": "COMPETES_WITH"
        }

    def extract_relationships(
        self,
        text: str
    ) -> List[Tuple[str, str, str]]:
        """
        Extract relationships using dependency parsing.

        Args:
            text: Input document text

        Returns:
            List of (subject, relation, object) triples

        Example:
            >>> extractor.extract_relationships("Elon Musk founded SpaceX")
            [("Elon Musk", "FOUNDED", "SpaceX")]
        """
        doc = self.nlp(text)
        relationships = []

        # Find verb-subject-object patterns in dependency parse
        for token in doc:
            if token.pos_ == "VERB":
                subject = self._find_subject(token)
                obj = self._find_object(token)

                if subject and obj:
                    relation = self._normalize_relation(token.lemma_)
                    relationships.append((subject, relation, obj))

        return relationships

    def _find_subject(self, verb_token):
        """Find subject of verb through dependency parse."""
        for child in verb_token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                return self._get_entity_span(child)
        return None

    def _find_object(self, verb_token):
        """Find object of verb through dependency parse."""
        for child in verb_token.children:
            if child.dep_ in ["dobj", "pobj", "attr"]:
                return self._get_entity_span(child)
        return None

    def _get_entity_span(self, token):
        """Extract full entity span from token, preferring named entities."""
        if token.ent_type_:
            return token.text
        else:
            # Build noun phrase from token subtree
            subtree = list(token.subtree)
            return " ".join([t.text for t in subtree])

    def _normalize_relation(self, verb: str) -> str:
        """
        Normalize verb to standardized relationship type.

        Args:
            verb: Verb lemma from dependency parse

        Returns:
            Standardized relationship type in UPPER_CASE
        """
        normalized = verb.lower().replace(" ", "_")
        return self.verb_mapping.get(normalized, normalized.upper())
