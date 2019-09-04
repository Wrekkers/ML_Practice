# Setup Guide

This guide will provide a step-by-step guide to make your machine ready to run Python with basic packages to get you started with machine learing.

## Installing Python

[Download PyCharm](https://www.jetbrains.com/pycharm/download/) (community version). It is one of the widely used IDE for Python and would make coding easier.

Install [HomeBrew](https://brew.sh/) a package manager for Mac OS by running the following commands in the terminal.
```bash
xcode-select --install
```
```bash
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Ensure Brew is correctly installed by running the following command.

```bash
brew doctor
```

Install Python 3.7 by running the following command.

```bash
brew install python3
```

After the installation check the version of the python and also its installation directory.

```bash
python3 --version
which python3
```

The ouput of the directory should be something like "/usr/local/bin/python3.7" or "/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7"

## Getting Started

1. Start a new project in PyCharm. And push the dropdown of "Project Interpreter".
2. Choose the option of "New environment using Virtualenv" and check that the version of the "Base interpreter" is Python3.7. Check your project directory and click "Create". [Virtualenv](https://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv) allows us to have a seprate environment for each project which allows for smooth development.
3. Create a new Python file either from the File menu or by right-clicking the project folder in the left widget.
4. Let's run a simple code to understand how to run and view output.
```python
if __name__ == '__main__':
	a = "This is a string."
	b = 10
	c = 55
	result = b + c
	print ("Hello World")
	print ("The sum of two numbers: {}".format(result))
```
5. To run the above program either click the small green triangle besides the "if \_\_name\_\_ == '\_\_main\_\_':" or run the program from the Run menu.
6. Feel free to explore some of the basic Python tutorials by clicking [here](https://www.datacamp.com/courses/intro-to-python-for-data-science).


## Solving our first ML Problem
We are gonna work on our first dataset by using simple [linear regression](http://onlinestatbook.com/2/regression/intro.html). We would be using a fairly small dataset for our problem called [Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg). Before we get stared we need to install some libraries in our Python environment. In order to do that, click on the terminal tab in PyCharm at the bottom of the window, it should have (venv) as its set environment. Run the following commands in it.
```bash
pip install pandas
pip install sklearn
pip install matplotlib
```
Our dataset is present in the file "auto-mpg.data", so download it inside your project directory.

### Preparing the Dataset
First let's see how can we explore the dataset in Python. Observe that the values of each row in the dataset are seperated by spaces which are not fixed in length. Thus it would require us to read the file line by line in Python. Below is the source code to do the needfull.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

headers = ['cylinders','displacement','horsepower','weight'
           ,'acceleration','model year','origin']
mpg = [] #Dependent variable : mpg (miles per gallon)
features = [] #Independent variables: rest of them except car name

#Preparing the data
raw = open('auto-mpg.data')
for line in raw:
    i=0
    var = []
    for y in line.split():
        if (i==0):  # The first column is mpg
            mpg.append([float(y)])
        if (i<8 and i>0):  # Not including the last column i.e. "car names"
            if(y=='?'): # Handaling missing values for "horsepower" 
                y = 0   # (Setting them to 0) Later we will change them
            if(i==4):
                var.append(float(y.replace('.','')))
            else:
                var.append(float(y))
        i+=1
    features.append(var)
    
 
df_x = pd.DataFrame.from_records(features, columns = headers)
df_y = pd.DataFrame.from_records(mpg, columns = ['mpg'])

```

Now we have the entire datset in the variables: '*df_x*' and '*df_y*'. You can preview these variables and get an overall summary of them by using:
```python
print(df_x.describe(include='all'))
print(df_y.describe(include='all'))
print(df_x.head()) # Prints the top 5 values
print(df_y.head()) # Prints the top 5 values
```
Now let us change the missing values, i.e. now 0, to their average values.
```python
avg_bhp = (np.average(df_x['horsepower'])*len(df_x))/(len(df_x)-6)
df_x['horsepower'] = df_x['horsepower'].replace(0,avg_bhp)
```
Now let's train our first ML model in Python.
```python
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
        df_x,df_y,test_size=0.2,random_state = 5)
lm = LinearRegression()

lm.fit(X_train, Y_train)
pred_test = lm.predict(X_test)

print('coefficients :(all vars)')
print(lm.coef_)

print('Intercept :(all vars)')
print(lm.intercept_)

print('Accuracy Score(all vars): %s' % lm.score(X_test,Y_test))

residues = (Y_test - pred_test)

plt.scatter(X_test, Y_test,  color='black') # The actual values
plt.plot(X_test, pred_test, color='blue', linewidth=3) # The predicted values
```
Voila! You have successfully completed your first ML model.
## License
This guide is free to use for non-commercial purposes. For any improvements please reach out to me on: bhanu93(dot)iitd(at)gmail(dot)com.
