# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## Reproduce the solution

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the reproducible training and prediction pipeline from the repository root:

```bash
python src/pipeline.py
```

This evaluates the median baseline, Ridge, and CatBoost on a January-August / September-October time split, retrains CatBoost on all labeled development data, and writes:

- `model_results.csv`
- `run_details.json`
- `validation_predictions.csv`
- `data/december-chart-inputs.csv` with predictions filled

The final validation file is populated from the supplied `data/validation-predictions-template.csv` so its required ID order is preserved.

Run the supplied scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

The scorer creates `scorer_results/candidate_december.png`. It validates the output structure and does not calculate the hidden final prediction metric.

See `Notebook.ipynb` for the executed analysis and `report.docx` for the assessment report.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation-predictions-template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
