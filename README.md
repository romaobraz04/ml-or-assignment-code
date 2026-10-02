# Machine Learning in OR assignment code

Python notebooks and supporting files for the 2026 Machine Learning in
Operations Research assignment.

## Files

- `src/regression_outliers.ipynb`: regression, PCA/PCR, Lasso, and outlier analysis.
- `src/data_2026.csv`: supplied numerical dataset used by the notebook.
- `src/lasso.ipynb` and `src/helpers.py`: currently empty placeholders.
- `requirements.txt`: Python analysis dependencies.

## Run the notebook

Install the dependencies and Jupyter:

```powershell
python -m pip install -r requirements.txt jupyter
cd src
python -m jupyter notebook
```

Open `regression_outliers.ipynb`. Start Jupyter from `src` so the notebook's
relative dataset and figure paths resolve correctly.

## Separate report repository

The local `6aad5eee42e0cccfd1864263/` directory contains the LaTeX report and
its independent Git repository connected to Overleaf. It is excluded from
this code repository, along with local backups, temporary files, the
assignment PDF, and generated figures. Run report Git commands from inside
that directory; run code Git commands from this repository's root.
