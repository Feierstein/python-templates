'''
from flask import Flask

# Create Flask application
app = Flask(__name__)

# Define route
@app.route('/')
def hello():
    return 'Hello, World!'

# WSGI entry point
if __name__ == '__main__':
    # Run Flask app using WSGI server
    from wsgiref.simple_server import make_server
    httpd = make_server('localhost', 8000, app)
    print("Serving on port 8000...")
    httpd.serve_forever()

'''

from flask import Flask
import sklearn_use_algorithm as algo

# Create a Flask app
app = Flask(__name__)

# Define a route
@app.route('/user_risk')

def get_algo():
    #return('calling algo.run_algo()')
    result = algo.run_algo_on_all()
    return result
 
@app.route('/hello')

def hello():
    return "hello"
    #result = algo.get_algo()
    #return result
#def run_algo():
 #   return 'Hello, World2!'

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
   
  

'''
 # working code

  from flask import Flask
import boto3_use_algorithm as algo

# Create a Flask app
app = Flask(__name__)

# Define a route
@app.route('/')

def hello():
    result = algo.get_algo()
    return result
    
#def run_algo():
 #   return 'Hello, World2!'

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
  '''