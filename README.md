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

The ouput of the directory should be somethhing like "/usr/local/bin/python3.7" or "/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7"

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
We are gonna work on our first dataset by using simple [linear regression](http://onlinestatbook.com/2/regression/intro.html). We would be using a fairly small dataset for our problem called [Auto MPG](https://archive.ics.uci.edu/ml/datasets/auto+mpg).

## License
This guide is free to use for non-commercial purposes. For any improvements please reach out to me on: bhanu93(dot)iitd(at)gmail(dot)com.
