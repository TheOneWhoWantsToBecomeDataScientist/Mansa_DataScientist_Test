# What's this ?

You'll find here my contribution to [Mansa's Kanedama DataScientist test](https://github.com/MansaGroup/kanedama/tree/main/datascience):
- `Data_Analysis.ipynb` file is a Jupyter Notebook that is dedicated to data analysis
- `Prediction_model.ipynb` file is a Jupyter Notebook that is dedicated to building and testing prediction models. 3 differents models have been tested.
  + Results of the third model fitting are stored in the `mdl_3_param.txt` file (JSON format), in order to reuse it later.
- `Account_synthesis.py` file is an implementation of Account_synthesis class, which includes properties and methods to facilitate next monthly outgoing prediction
- `main.py` file implements the prediction model API using FastAPI (run `uvicorn main:app` in the working directory)
- `test_main.py` file can be used to test the API (run it if `uvicorn main:app` has been launched in other terminal)
  
  
# Environment  
  
This work has been developped with `Python` 3.7, using `Anaconda` distribution.
To reproduce a working environment, you'll find different files:
- Using Anaconda
  + `environment.yml` file contains the installed package using conda
  + As conda automatically installs a lot of dependencies for each package, and according to your platform, the full list of dependencies is given in `environment_full.yml` (for Windows platform). Installing packages from `environment.yml` file will adapt this dependencies to your platform.
- Using Pip, this should be equivalent to these requirements files:
  + `requirements.txt` file contains the required packages to run the API 
  + `requirements-eda.txt` file contains the required packages to run the API and notebooks
  