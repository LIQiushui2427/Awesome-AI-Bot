check:
	@echo "Check format of all files..."
	black --check .

fix:
	@echo "Format all files..."
	black .
