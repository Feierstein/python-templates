from flask import Flask
from flask_basicauth import BasicAuth

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'username'
app.config['BASIC_AUTH_PASSWORD'] = 'password'

basic_auth = BasicAuth(app)

@app.route('/secure-endpoint')
@basic_auth.required
def secure_endpoint():
    # Authorized users can access this endpoint
    return 'This is a secure endpoint'

if __name__ == '__main__':
    app.run(debug=True)