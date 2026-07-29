.PHONY: help validate video-catalog precommit inventory

.DEFAULT_GOAL := validate

help:
	@printf '%s\n' \
	  'make validate       Validate first-party repository files and the video catalog' \
	  'make video-catalog  Regenerate video catalog and review-queue files' \
	  'make precommit      Run all configured pre-commit hooks' \
	  'make inventory      Write a SHA-256 inventory of References/'

validate:
	python3 scripts/validate_repository.py
	python3 videos/validate_video_resources.py --check-generated

video-catalog:
	python3 videos/validate_video_resources.py --write

precommit:
	pre-commit run --all-files

inventory:
	python3 scripts/build_reference_inventory.py References/reference_inventory.csv
