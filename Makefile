submission:
	python src/scripts/mask_to_submission.py

move_data:
	python src/preprocessing/move_datasets.py

preprocess_data:
	python src/preprocessing/preprocessing.py

preprocessing: move_data preprocess_data