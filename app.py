# app.py
import cv2
import os
import json
from flask import Flask, request, render_template, Response, redirect, url_for, flash, jsonify, session
from datetime import date, datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash # Import for hashing new registration passwords

# Import database models
from models import db, User, Attendance

# Initialize Flask App and Login Manager
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a_very_secret_key_that_is_long_and_secure'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect to 'login' view if user is not logged in

# --- Constants ---
N_IMGS = 10
FACE_DETECTOR = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --- Flask-Login User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Helper Functions ---
def create_initial_admin():
    """Creates a default admin user if one doesn't exist."""
    with app.app_context():
        if not User.query.filter_by(role='staff').first():
            admin_user = User(id=1, username='admin', role='staff')
            admin_user.set_password('admin123') # Hashed password
            db.session.add(admin_user)
            db.session.commit()
            print("Default admin created with username 'admin', ID '1' and password 'admin123'")

def train_model():
    """Train the face recognition model with images from the database."""
    faces = []
    labels = []
    
    users = User.query.all()
    if not users:
        # No flash message here, as it might run in background
        print("No users in the database to train the model.")
        return

    for user in users:
        user_folder = f'static/faces/{user.username}_{user.id}'
        if os.path.isdir(user_folder):
            for imgname in os.listdir(user_folder):
                img_path = os.path.join(user_folder, imgname)
                img = cv2.imread(img_path)
                if img is not None:
                    resized_face = cv2.resize(img, (50, 50))
                    faces.append(resized_face.ravel())
                    labels.append(f'{user.username}_{user.id}')
    
    if not faces:
        print("No face images found to train the model. Training skipped.")
        return
        
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')
    print("Model trained successfully!") # Use print for background tasks

def extract_faces(img):
    """Extract face coordinates from an image."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = FACE_DETECTOR.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

# --- Video Streaming Generators (Same as before) ---
def gen_frames_add_user(userid, username):
    cap = cv2.VideoCapture(0)
    img_count = 0
    user_folder = f'static/faces/{username}_{userid}'
    if not os.path.isdir(user_folder):
        os.makedirs(user_folder)

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {img_count}/{N_IMGS}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            
            if img_count < N_IMGS:
                if len(faces) > 0 and (datetime.now().microsecond % 10 == 0):
                    face_img = frame[y:y+h, x:x+w]
                    img_path = os.path.join(user_folder, f'{username}_{img_count}.jpg')
                    cv2.imwrite(img_path, face_img)
                    img_count += 1
        
        if img_count >= N_IMGS:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()
    print(f"Face capture complete for {username} (ID: {userid})")


def gen_frames_mark_attendance(current_user_id):
    """Video streaming generator for marking attendance for a specific user."""
    try:
        model = joblib.load('static/face_recognition_model.pkl')
    except FileNotFoundError:
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + b'Model not found. Please add a user and train the model.' + b'\r\n')
        return

    cap = cv2.VideoCapture(0)
    attendance_marked = False # Flag to stop attendance marking after first success
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face_img = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person_str = model.predict(face_img.reshape(1, -1))[0]
            
            # Extract ID from model prediction
            try:
                identified_userid = int(identified_person_str.split('_')[1])
            except (ValueError, IndexError):
                identified_userid = -1 # Invalid format

            # Check if identified face matches the logged-in user and mark attendance
            if identified_userid == current_user_id and not attendance_marked:
                with app.app_context():
                    user = User.query.get(current_user_id)
                    if user:
                        today = date.today()
                        existing_attendance = Attendance.query.filter(
                            func.date(Attendance.timestamp) == today, 
                            Attendance.user_id == user.id
                        ).first()
                        if not existing_attendance:
                            new_attendance = Attendance(user_id=user.id)
                            db.session.add(new_attendance)
                            db.session.commit()
                            attendance_marked = True # Mark as true to prevent duplicate entries
                            flash("Attendance marked successfully!", "success")
                        else:
                            flash("Attendance already marked for today.", "info")
                    else:
                        flash("User not found in database.", "danger")
                
            display_text = f'{identified_person_str}'
            if attendance_marked:
                display_text += " (ATTENDED)"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            cv2.putText(frame, display_text, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\r\n')
    
    cap.release()
    print("Attendance marking stream ended.")

# --- Authentication & Registration Routes ---
@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        user_id = request.form['userid']
        password = request.form['password']
        user = User.query.get(user_id)
        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully.', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid ID or password. Please try again.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        userid = request.form['userid']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role = request.form['role']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('register.html', userid=userid, username=username, role=role)

        if User.query.get(userid):
            flash(f'User with ID {userid} already exists.', 'danger')
            return render_template('register.html', userid=userid, username=username, role=role)
        
        if User.query.filter_by(username=username).first():
            flash(f'Username {username} is already taken.', 'danger')
            return render_template('register.html', userid=userid, username=username, role=role)

        new_user = User(id=userid, username=username, role=role)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash(f'Account created for {username}. Please log in. If you are a new user, you must register your face by going to the dashboard.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# --- Main Dashboard (Redirects based on role) ---
@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'staff':
        return redirect(url_for('admin_dashboard'))
    else:
        return redirect(url_for('student_dashboard'))

# --- Admin (Staff) Routes ---
@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'staff':
        flash('Access denied. Staff role required.', 'danger')
        return redirect(url_for('dashboard'))
    
    total_users = User.query.count()
    today = date.today()
    attendance_today_count = Attendance.query.filter(func.date(Attendance.timestamp) == today).count()
    all_attendance_records = Attendance.query.order_by(Attendance.timestamp.desc()).all()
    all_users = User.query.all()
    
    return render_template('admin_dashboard.html', 
                           total_users=total_users, 
                           attendance_today_count=attendance_today_count,
                           all_attendance_records=all_attendance_records,
                           all_users=all_users)

@app.route('/admin/add_user_details', methods=['POST'])
@login_required
def admin_add_user_details():
    if current_user.role != 'staff':
        flash('Access denied. Staff role required.', 'danger')
        return redirect(url_for('dashboard'))
    
    userid = request.form['userid']
    username = request.form['username']
    password = request.form['password']
    role = request.form['role']

    if User.query.get(userid):
        flash(f'User with ID {userid} already exists.', 'danger')
        return redirect(url_for('admin_dashboard'))
    if User.query.filter_by(username=username).first():
        flash(f'Username {username} is already taken.', 'danger')
        return redirect(url_for('admin_dashboard'))

    new_user = User(id=userid, username=username, role=role)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    flash(f'User {username} (ID: {userid}) added. Now proceed to capture their face.', 'success')
    return redirect(url_for('capture_face', userid=new_user.id))

@app.route('/delete_user/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role != 'staff':
        flash('Access denied. Staff role required.', 'danger')
        return redirect(url_for('dashboard'))

    user = User.query.get_or_404(user_id)
    
    # Delete the user's face images folder if it exists
    user_folder = f'static/faces/{user.username}_{user.id}'
    if os.path.isdir(user_folder):
        import shutil
        shutil.rmtree(user_folder)
        
    db.session.delete(user)
    db.session.commit()
    flash(f'User {user.username} has been deleted. Retraining model...', 'success')
    train_model() # Retrain model after deleting user
    return redirect(url_for('admin_dashboard'))

@app.route('/train_model_route', methods=['GET'])
@login_required
def train_model_route():
    if current_user.role != 'staff':
        flash('Access denied. Staff role required.', 'danger')
        return redirect(url_for('dashboard'))
    train_model()
    flash('Model retraining initiated.', 'info')
    return redirect(url_for('admin_dashboard')) # Redirect to admin dashboard after training

# --- Student Routes ---
@app.route('/student/dashboard')
@login_required
def student_dashboard():
    if current_user.role != 'student':
        flash('Access denied. Student role required.', 'danger')
        return redirect(url_for('dashboard'))
    
    today = date.today()
    attendance_today = Attendance.query.filter(
        func.date(Attendance.timestamp) == today,
        Attendance.user_id == current_user.id
    ).first()
    
    my_attendance_history = Attendance.query.filter_by(user_id=current_user.id).order_by(Attendance.timestamp.desc()).all()

    return render_template('student_dashboard.html', 
                           attendance_today=attendance_today, 
                           history=my_attendance_history)

# --- Common Face Recognition & Attendance Routes ---

@app.route('/mark_attendance')
@login_required
def mark_attendance():
    return render_template('mark_attendance.html')

@app.route('/video_feed/mark_attendance')
@login_required
def video_feed_mark_attendance():
    # Pass current_user.id to the generator to ensure it only marks for the logged-in user
    return Response(gen_frames_mark_attendance(current_user.id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_face/<int:userid>')
@login_required
def capture_face(userid):
    if current_user.role != 'staff': # Only staff can initiate face capture for new users
        flash('Access denied. Staff role required to capture faces.', 'danger')
        return redirect(url_for('dashboard'))
    user = User.query.get_or_404(userid)
    if os.path.isdir(f'static/faces/{user.username}_{user.id}'):
        flash(f"Face images for {user.username} already exist. If you want to recapture, delete the folder first manually (or add a delete button for faces).", "warning")
        # return redirect(url_for('admin_dashboard')) # Prevent recapture unless desired
    return render_template('capture_face.html', user=user, n_imgs=N_IMGS)

@app.route('/video_feed/capture_face/<int:userid>/<username>')
@login_required
def video_feed_capture_face(userid, username):
    if current_user.role != 'staff': # Ensure only staff can use this
        return Response(b'', mimetype='multipart/x-mixed-replace; boundary=frame') # Empty response
    return Response(gen_frames_add_user(userid, username), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API for Chart Data ---
@app.route('/api/attendance_chart_data')
@login_required
def attendance_chart_data():
    if current_user.role != 'staff':
        return jsonify({'error': 'Unauthorized'}), 403
    
    today = date.today()
    last_7_days = [today - timedelta(days=i) for i in range(6, -1, -1)]
    data = []
    for day in last_7_days:
        count = Attendance.query.filter(func.date(Attendance.timestamp) == day).count()
        data.append({'date': day.strftime('%b %d'), 'count': count})
        
    return jsonify(data)

# --- Main Execution ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    create_initial_admin() # Ensure an admin exists
    app.run(debug=True)