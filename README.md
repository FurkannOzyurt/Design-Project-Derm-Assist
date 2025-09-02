# Dermatology Assistant Web Application

A web-based dermatology assistant that combines image classification and natural language processing to provide medical insights and answers to user queries.

## Features

- User authentication (login/register)
- Image upload and classification
- Chat interface with message history
- Integration with ConvNeXt-Tiny classifier
- Integration with Mistral-7B-Instruct LLM
- Retrieval-Augmented Generation (RAG) for enhanced responses

## Prerequisites

- Python 3.8+
- PyTorch
- Flask
- SQLite (or PostgreSQL)
- Your trained ConvNeXt-Tiny model
- Your Mistral-7B-Instruct model
- Your RAG implementation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dermatology-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export SECRET_KEY="your-secret-key-here"  # On Windows: set SECRET_KEY=your-secret-key-here
```

5. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

## Configuration

1. Update the model paths in `models.py`:
   - Set the path to your trained ConvNeXt-Tiny model
   - Set the path to your Mistral-7B-Instruct model
   - Configure your RAG implementation

2. (Optional) Configure PostgreSQL:
   - Update `SQLALCHEMY_DATABASE_URI` in `app.py` with your PostgreSQL connection string

## Running the Application

1. Start the Flask development server:
```bash
python app.py
```

2. Access the application at `http://localhost:5000`

## Usage

1. Register a new account or log in
2. Start a new chat
3. Upload a dermatology image with your first message
4. Continue the conversation with text-only messages
5. View chat history in the left panel

## Project Structure

```
dermatology-assistant/
├── app.py              # Main Flask application
├── models.py           # Model integration (classifier, LLM, RAG)
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   └── chat.html
├── static/            # Static files (CSS, JS)
├── uploads/           # Uploaded images
└── instance/          # Database files
```

## Security Considerations

- All user passwords are hashed using Werkzeug's security functions
- File uploads are restricted to image files
- Maximum file size is limited to 16MB
- User authentication is required for all sensitive routes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 