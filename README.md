# AI Agent for Sepsis EHR Analysis 

## Project Notes

* Develop individual tools first (e.g., retrieve info of a patient from user's query, collect simple stats, missing values imputation, train model to do mortality prediction, etc.) -> they'll need to work together later on when called by the agent
* Add small noise (e.g., Gaussian noise) to input, observe if imputation result changes a lot (e.g., by evaluating the statistical properties of each feature) -> evaluate the quality of the imputation results
* Develop a dataset of healthy people (question: how do we define healthy/unhealthy from the given attributes of a patient? i.e., what is the ground truth?)
