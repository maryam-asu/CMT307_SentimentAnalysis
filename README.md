# CMT307

This goal of this project is to perform sentiment analysis for IMDb reviews. It is part of a course project in Cardiff School of Computer Science and Informatics. 

## Getting Started

To download a copy of the project, use the following command. 
```
git clone https://github.com/maryam-asu/CMT307-SentimentAnalysis.git
```


### Prerequisites

Python has to be installed in your system. In addition, you need to install some packages listed in the requirements.txt. We recommend you install them in a new Python environment. 
To install the packages using pip.
```
pip install -r requirements.txt
```
To install the packages using Conda.
```
conda install --yes --file requirements.tx
```


### Running the code

The program accepts three arguments (train set, test set, and development set) and outputs the result on the screen. The directory of each set should be passed; otherwise, the default path is used (e.g., datasets/IMDb/train/). Example
```
python sentiment_analysis.py --input-train-file datasets/IMDb/train --input-te
st-file datasets/IMDb/test --input-dev-file datasets/IMDb/dev
```
Expected output

```
Precision=0.8857818038597873 : Recall=0.8999599839935974 : FScore=0.8928146089718142 : Accuracy=0.892
```

End with an example of getting some data out of the system or using it for a little demo
