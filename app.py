from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from models import DermatologyAssistant
import sqlite3

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dermatology_assistant.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize the assistant
dermatology_assistant = DermatologyAssistant()

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    chats = db.relationship('Chat', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages = db.relationship('Message', backref='chat', lazy=True)
    image_path = db.Column(db.String(255), nullable=True)
    classification_result = db.Column(db.String(255), nullable=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

def init_db():
    with app.app_context():
        # Drop all tables and recreate them
        db.drop_all()
        db.create_all()
        print("Database initialized successfully")

# Initialize the database
init_db()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('chat'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('chat'))
        
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/chat')
@login_required
def chat():
    chat_id = request.args.get('chat_id')
    chats = Chat.query.filter_by(user_id=current_user.id).order_by(Chat.created_at.desc()).all()
    return render_template('chat.html', chats=chats, current_chat_id=int(chat_id) if chat_id else None)

@app.route('/api/chats')
@login_required
def get_chats():
    if not current_user.is_authenticated:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        chats = db.session.query(Chat).filter_by(user_id=current_user.id).order_by(Chat.created_at.desc()).all()
        return jsonify([{
            'id': chat.id,
            'title': chat.classification_result or 'New Chat',
            'created_at': chat.created_at.isoformat(),
            'image_path': chat.image_path
        } for chat in chats])
    except Exception as e:
        print(f"Error in get_chats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/messages/<int:chat_id>')
@login_required
def get_messages(chat_id):
    if not current_user.is_authenticated:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        chat = db.session.get(Chat, chat_id)
        if not chat or chat.user_id != current_user.id:
            return jsonify({'error': 'Chat not found'}), 404
        
        messages = db.session.query(Message).filter_by(chat_id=chat_id).order_by(Message.created_at.asc()).all()
        return jsonify([{
            'id': message.id,
            'content': message.content,
            'is_user': message.is_user,
            'created_at': message.created_at.isoformat()
        } for message in messages])
    except Exception as e:
        print(f"Error in get_messages: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@login_required
def create_chat():
    if not current_user.is_authenticated:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        chat = Chat(user_id=current_user.id)
        db.session.add(chat)
        db.session.commit()
        return jsonify({'chat_id': chat.id})
    except Exception as e:
        print(f"Error in create_chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/message', methods=['POST'])
@login_required
def send_message():
    if not current_user.is_authenticated:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        chat_id = request.form.get('chat_id')
        message = request.form.get('message', '')
        image = request.files.get('image')
        
        chat = db.session.get(Chat, chat_id)
        if not chat or chat.user_id != current_user.id:
            return jsonify({'error': 'Chat not found'}), 404
        
        # Check if chat already has an image
        has_image = bool(chat.image_path)
        
        # Process image if provided and chat doesn't have one
        if image and image.filename and not has_image:
            # Save image
            filename = secure_filename(image.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            
            # Classify image
            result = dermatology_assistant.predict_image(image_path)
            
            # Update chat with image path and classification
            chat.image_path = filename
            chat.classification_result = result
            db.session.commit()
            
            # Get RAG context
            rag_context = dermatology_assistant.get_rag_context(result, result)
            
            # Create image message
            image_message = Message(
                chat_id=chat.id,
                content=f'<img src="/uploads/{filename}" alt="Uploaded image">',
                is_user=True
            )
            db.session.add(image_message)
        else:
            rag_context = None
        
        # Create text message if there is one
        if message:
            text_message = Message(
                chat_id=chat.id,
                content=message,
                is_user=True
            )
            db.session.add(text_message)
        
        # Generate assistant response
        assistant_response = dermatology_assistant.generate_response(
            image_class=chat.classification_result,
            user_message=message,
            rag_context=rag_context
        )
        
        # Create assistant message
        assistant_message = Message(
            chat_id=chat.id,
            content=assistant_response,
            is_user=False
        )
        db.session.add(assistant_message)
        db.session.commit()
        
        return jsonify({
            'user_message': {
                'content': message,
                'timestamp': text_message.created_at.isoformat() if message else None
            },
            'assistant_message': {
                'content': assistant_response,
                'timestamp': assistant_message.created_at.isoformat()
            }
        })
        
    except Exception as e:
        print(f"Error in send_message: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/chat/<int:chat_id>/messages')
@login_required
def get_chat_messages(chat_id):
    messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.created_at).all()
    return jsonify([{
        'id': msg.id,
        'content': msg.content,
        'is_user': msg.is_user,
        'image_path': msg.image_path,
        'created_at': msg.created_at.isoformat()
    } for msg in messages])

@app.route('/api/chat/<int:chat_id>')
@login_required
def get_chat(chat_id):
    if not current_user.is_authenticated:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        chat = db.session.get(Chat, chat_id)
        if not chat or chat.user_id != current_user.id:
            return jsonify({'error': 'Chat not found'}), 404
        
        return jsonify({
            'id': chat.id,
            'title': chat.classification_result or 'New Chat',
            'created_at': chat.created_at.isoformat(),
            'image_path': chat.image_path,
            'classification_result': chat.classification_result
        })
    except Exception as e:
        print(f"Error in get_chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 