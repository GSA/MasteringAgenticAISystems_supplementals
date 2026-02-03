    def extract_from_api(
        self,
        endpoint: str,
        auth_token: str,
        since: datetime = None
    ) -> List[Dict[str, Any]]:
        """
        Extract documents from REST API with pagination and rate limiting.

        This method demonstrates robust API extraction handling common
        challenges: paginated responses, rate limits, and network failures.

        Args:
            endpoint: API endpoint URL (e.g., Zendesk articles endpoint)
            auth_token: Bearer token for authentication
            since: Extract only articles updated after this timestamp

        Returns:
            List of article dictionaries from API responses
        """
        logger.info(f"Extracting from API: {endpoint}")

        documents = []
        page = 1
        per_page = 100

        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }

        while True:
            # Build request with pagination parameters
            params = {
                "page": page,
                "per_page": per_page
            }

            if since:
                params["updated_since"] = since.isoformat()

            try:
                response = requests.get(
                    endpoint,
                    headers=headers,
                    params=params,
                    timeout=30
                )

                response.raise_for_status()

                data = response.json()

                # Extract articles from response structure
                articles = data.get("articles", [])

                if not articles:
                    break  # No more pages

                documents.extend(articles)

                logger.info(f"  Page {page}: {len(articles)} documents")

                # Check if we've reached the last page
                if len(articles) < per_page:
                    break

                page += 1

                # Rate limiting (if needed)
                # time.sleep(60 / rate_limit)

            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed: {e}")
                break

        logger.info(f"✓ Extracted {len(documents)} documents from API")

        return documents
