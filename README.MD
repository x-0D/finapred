![Mascot image](./finapred.png)

![License](https://badgen.net/badge/license/MIT?color=green)
![License](https://badgen.net/badge/UwU/furry%20product?color=orange)

# Finapred - Finance ~~Predator~~ Predictor

Finapred is a finance predictor app that uses fbprophet, binance api and streamlit framework to predict stock forecast. It provides users with an easy-to-use interface to make predictions on stocks and other financial instruments. Finapred is designed to help investors make informed decisions by providing accurate predictions on the future performance of stocks and other financial instruments.

## About
Finapred is a powerful tool for predicting stock prices. It uses fbprophet, binance api and streamlit framework to provide accurate predictions of stock prices. The app allows users to view historical data and make predictions about future stock prices. Finapred also provides users with the ability to customize their predictions by adjusting parameters such as the number of days in the forecast period and the confidence interval. 

## Installation

### Prerequisites 
Before you begin, make sure you have the following installed: 
- Python 3.8 or higher 
- Pipenv
- Docker
- Heroku CLI (optional)

### Step 1: Clone the Repository 
Clone the Finapred repository to your local machine: 
```bash 
git clone https://github.com/x-0D/finapred.git 
```  

### Step 2: Install Dependencies  
Navigate to the cloned repo and install the dependencies specified in the requirements.txt file using pipenv:  
```bash  
pipenv install -r requirements.txt  
```  

### Step 3: Run the App Locally   
You can now run the app locally using Streamlit:   
```bash   
streamlit run app.py   
```   

## Deployment    

Finapred can be deployed using either Heroku or Docker.    

### Heroku Deployment    

To deploy Finapred on Heroku, you will need to create a new Heroku app and push your code to it using Git. You can find detailed instructions on how to do this [here](https://devcenter.heroku.com/articles/git). Once your code is pushed, you will need to set up a Procfile which tells Heroku how to run your application. The Procfile for Finapred is already included in this repository so all you need to do is push it up with your code and then deploy your application on Heroku.     

### Docker Deployment     

To deploy Finapred with Docker, first build an image of your application using the provided Dockerfile:     
```bash
docker build -t finapred .
```

Then run it with docker-compose:
```bash
docker-compose up
```
This will start a container running Finapred on port 8501 which can be accessed from http://localhost:8501/ 

## Usage
Using Finapred is simple: users can select their desired stock or financial instrument and enter the desired time frame for prediction. The app will then generate a forecast based on historical data and current market conditions. Users can also customize their forecasts by adjusting parameters such as seasonality and trend components. 


## Denial of Responsibility 
Finapred is provided “as is” without any warranties or guarantees of any kind either expressed or implied including but not limited to accuracy or reliability of results obtained from its use. The user assumes all risks associated with its use including but not limited to financial losses due to inaccurate predictions made by Finapred’s algorithms or other unforeseen circumstances related to its use. 

<!--## Contributing 
We welcome contributions from anyone interested in helping improve Finapred! If you would like to contribute please fork this repository and submit a pull request with your changes for review by our team before merging into master branch. Please read our contributing guidelines before submitting your pull request: https://github.com/x-0D/finapred/contributing-guidelines -->

## License 
Finapred is released under the MIT license which can be found in the LICENSE file in this repository: https://github.com/x-0D/finapred/LICENSE
