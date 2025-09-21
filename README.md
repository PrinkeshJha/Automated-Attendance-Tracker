

<!-- ````markdown -->
# 📸 Automated Attendance Tracker  

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Framework-black?logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A **full-stack web application** that automates attendance using **real-time facial recognition**.  
This system ensures **accuracy, security, and efficiency**, with **role-based access** for students and staff.  

---


## ❌ The Problem  

Traditional attendance methods are:  
- Time-consuming ⏳  
- Error-prone ❌  
- Vulnerable to **proxy attendance** 🎭  

Managing physical records is tedious, and analyzing attendance trends is manual and inefficient.  

✅ **Our Solution:** Automate attendance with **facial recognition** — fast, reliable, and digital.  

---

## ✨ Key Features  

- 🔐 **Role-Based Access Control (RBAC)** – Separate dashboards for **Students** and **Staff**.  
- 👤 **Secure Authentication & Registration** – Unique IDs & hashed passwords.  
- 📸 **Real-Time Facial Recognition** – Students mark attendance with the camera.  
- 📊 **Interactive Admin Dashboard** – Includes:  
  - Attendance stats & 7-day trends (Chart.js).  
  - User management (Add/Delete/Capture Faces).  
  - Attendance logs.  
- 🎓 **Student Portal** – View attendance history & mark daily presence.  
- 🧠 **On-the-Fly Model Training** – Retrain recognition model after adding new users.  

---

## 🛠 Tech Stack & Architecture  

| Component         | Technology                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| **Backend**       | [Flask](https://flask.palletsprojects.com/)                                |
| **Database ORM**  | [Flask-SQLAlchemy](https://flask-sqlalchemy.palletsprojects.com/)           |
| **Authentication**| [Flask-Login](https://flask-login.readthedocs.io/)                         |
| **Database**      | [SQLite](https://www.sqlite.org/)                                          |
| **Computer Vision**| [OpenCV](https://opencv.org/)                                             |
| **ML Model**      | [Scikit-learn](https://scikit-learn.org/) (k-NN Classifier)                 |
| **Frontend**      | HTML, CSS, JavaScript, [Bootstrap 5](https://getbootstrap.com/)             |
| **Charts**        | [Chart.js](https://www.chartjs.org/)                                       |

---

## 🚀 Getting Started  

### ✅ Prerequisites  
- Python 3.8+  
- `pip` (Python package manager)  
- A working **webcam**  

### 🔧 Installation  

1. **Clone the repo**  
```bash
git clone https://github.com/YOUR_USERNAME/automated-attendance-tracker.git
cd automated-attendance-tracker
```

2. **Create & activate a virtual environment**

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download Haar Cascade for face detection**
   Download [`haarcascade_frontalface_default.xml`](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml)
   Place it in the **project root directory**.

5. **Run the application**

```bash
python app.py
```

Visit 👉 `http://127.0.0.1:5000` in your browser.

---

## 📋 Usage Guide

### 🔑 Default Admin Account

* **Username:** `admin`
* **Password:** `admin123`

### 👨‍🏫 Admin Workflow

1. Log in with admin credentials.
2. Add new users (students or staff).
3. Capture their face using the **"Capture Face"** button.
4. Retrain the recognition model with **"Retrain Model"**.

### 🎓 Student Workflow

1. Register via the **Register Page**.
2. Admin captures their face & retrains model.
3. Student logs in → clicks **"Mark Today's Attendance"** → looks at camera → ✅ attendance logged.

---

## 📂 Project Structure

```
/Smart-Facial-Attendance
|
|-- app.py
|-- models.py
|-- haarcascade_frontalface_default.xml
|-- requirements.txt
|
|-- instance/
|   |-- attendance.db  (Created automatically)
|
|-- static/
|   |-- faces/
|   |-- face_recognition_model.pkl
|
|-- templates/
|   |-- _base.html
|   |-- login.html
|   |-- admin_dashboard.html
|   |-- student_dashboard.html
|   |-- mark_attendance.html
|   |-- capture_face.html
```
---

## 🔮 Future Scope

* 🤖 **Deep Learning Models** (FaceNet, Siamese Networks).
* 🛡 **Liveness Detection** (anti-spoofing).
* ⚡ **Asynchronous Tasks** (Celery for background training).
* 📊 **Detailed Reporting** (Export to PDF/Excel).
* 🐳 **Docker Support** for deployment.

---

## 📄 License

Licensed under the **MIT License**.

```
© 2025 PrinkeshJha
```

---

## 🙌 Credits

* Built with ❤️ using **Flask + OpenCV + Scikit-learn**
* Frontend powered by **Bootstrap 5 + Chart.js**

<!-- ``` -->

