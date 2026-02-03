    def _extract_entities_from_docs(
        self,
        docs: List[Any]
    ) -> List[str]:
        """
        Extract named entities from documents for graph linking.

        This is the critical bridge between unstructured retrieval
        (vector search returns documents) and structured reasoning
        (graph queries need specific entity starting points).

        Args:
            docs: Retrieved documents from vector search

        Returns:
            Entity names for graph query starting points
        """
        entities = set()

        for doc in docs:
            # Use NER to identify entity mentions in text
            spacy_doc = self.nlp(doc.page_content)

            for ent in spacy_doc.ents:
                # Filter to entities likely in our knowledge graph
                if ent.label_ in ["PERSON", "ORG", "GPE"]:
                    entities.add(ent.text)

        return list(entities)
