import os
from web_app import create_app

app = create_app()

if __name__ == '__main__':
    # Use Heroku's port if available, otherwise default to 5000 for local dev
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)