import cv21
import numpy as np
import json
import os
from datetime import datetime
import pickle
class FaceAttendanceSystem:
    def __init__(self):
        self.employee_faces = {}  # {name: face_data}
        self.attendance_records = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.data_file = "face_data.pkl"
        self.attendance_file = "attendance_records.json"
        self.trained = False
        self.load_data()
    def load_data(self):
        """Load saved face data and attendance records"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                self.employee_faces = data['faces']
                if data['faces']:
                    self.train_recognizer()
            print(f"Loaded {len(self.employee_faces)} employees")
        if os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'r') as f:
                self.attendance_records = json.load(f)
    def save_data(self):
        """Save face data and attendance records"""
        with open(self.data_file, 'wb') as f:
            pickle.dump({'faces': self.employee_faces}, f)
        
        with open(self.attendance_file, 'w') as f:
            json.dump(self.attendance_records, f, indent=4)
    def train_recognizer(self):
        """Train the face recognizer with all employee faces"""
        if not self.employee_faces:
            return
        faces = []
        labels = []
        name_to_id = {name: idx for idx, name in enumerate(self.employee_faces.keys())}
        for name, face_samples in self.employee_faces.items():
            for face in face_samples:
                faces.append(face)
                labels.append(name_to_id[name])
        if faces:
            self.recognizer.train(faces, np.array(labels))
            self.trained = True
            self.name_to_id = name_to_id
            self.id_to_name = {v: k for k, v in name_to_id.items()}
    def register_employee(self, name):
        """Register a new employee by capturing their face from camera"""
        if name in self.employee_faces:
            print(f"Employee '{name}' already registered!")
            overwrite = input("Do you want to re-register? (yes/no): ").lower()
            if overwrite != 'yes':
                return False
        print(f"\nRegistering employee: {name}")
        print("Position your face in front of the camera...")
        print("We'll capture multiple angles. Press SPACE to capture, ESC to finish.")
        video_capture = cv2.VideoCapture(0)
        face_samples = []
        count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Samples: {count}/30", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to capture", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Register Employee - ESC to finish', frame)
            key = cv2.waitKey(1)
            if key == 32 and len(faces) > 0:  # SPACE key
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                face_samples.append(face_roi)
                count += 1
                print(f"Captured sample {count}/30")
                if count >= 30:
                    break
            elif key == 27:  # ESC key
                break
        video_capture.release()
        cv2.destroyAllWindows()
        if len(face_samples) < 10:
            print(f"Not enough samples captured ({len(face_samples)}). Need at least 10.")
            return False
        self.employee_faces[name] = face_samples
        self.attendance_records[name] = []
        self.train_recognizer()
        self.save_data()
        print(f"\n Employee '{name}' registered successfully with {len(face_samples)} samples!")
        return True
    def mark_attendance(self, name, action='check_in'):
        """Mark check-in or check-out for an employee"""
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        date = now.strftime("%Y-%m-%d")
        if name not in self.attendance_records:
            self.attendance_records[name] = []
        today_records = [r for r in self.attendance_records[name] if r['date'] == date]
        if action == 'check_in':
            if any(r['action'] == 'check_in' and 'check_out_time' not in r for r in today_records):
                print(f"{name} already checked in today!")
                return False
            record = {
                'date': date,
                'action': 'check_in',
                'check_in_time': timestamp
            }
            self.attendance_records[name].append(record)
            print(f"\n{'='*50}")
            print(f"✓ CHECK-IN SUCCESSFUL")
            print(f"  Employee: {name}")
            print(f"  Time: {now.strftime('%H:%M:%S')}")
            print(f"  Date: {date}")
            print(f"{'='*50}\n")
        elif action == 'check_out':
            open_record = None
            for r in reversed(today_records):
                if r['action'] == 'check_in' and 'check_out_time' not in r:
                    open_record = r
                    break
            if not open_record:
                print(f"{name} hasn't checked in yet today!")
                return False
            open_record['check_out_time'] = timestamp
            open_record['action'] = 'completed'
            check_in = datetime.strptime(open_record['check_in_time'], "%Y-%m-%d %H:%M:%S")
            check_out = datetime.strptime(open_record['check_out_time'], "%Y-%m-%d %H:%M:%S")
            duration = check_out - check_in
            hours = duration.total_seconds() / 3600
            open_record['hours_worked'] = round(hours, 2)
            print(f"\n{'='*50}")
            print(f"✓ CHECK-OUT SUCCESSFUL")
            print(f"  Employee: {name}")
            print(f"  Time: {now.strftime('%H:%M:%S')}")
            print(f"  Hours Worked Today: {open_record['hours_worked']} hours")
            print(f"{'='*50}\n")
        self.save_data()
        return True
    def recognize_face_from_camera(self, action='check_in'):
        """Open camera and recognize face for attendance"""
        if not self.trained:
            print("No employees registered yet!")
            return None
        video_capture = cv2.VideoCapture(0)
        print(f"\nCamera opened for {action.upper()}")
        print("Position your face in front of the camera...")
        print("Press 'q' to quit\n")
        recognized_name = None
        confidence_threshold = 70
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                label, confidence = self.recognizer.predict(face_roi)
                if confidence < confidence_threshold:
                    name = self.id_to_name[label]
                    color = (0, 255, 0)
                    text = f"{name} ({int(100-confidence)}%)"
                    if recognized_name != name:
                        recognized_name = name
                        self.mark_attendance(name, action)
                        video_capture.release()
                        cv2.destroyAllWindows()
                        return name
                else:
                    name = "Unknown"
                    color = (0, 0, 255)
                    text = "Unknown"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Action: {action.upper()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('Face Recognition Attendance', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.destroyAllWindows()
        return None
    def get_employee_report(self, name):
        """Generate attendance report for an employee"""
        if name not in self.attendance_records:
            print(f"Employee '{name}' not found!")
            return
        records = self.attendance_records[name]
        print(f"\n{'='*70}")
        print(f"ATTENDANCE REPORT: {name}")
        print(f"{'='*70}")
        total_hours = 0
        total_days = 0
        for record in records:
            if record['action'] == 'completed':
                total_days += 1
                total_hours += record['hours_worked']
                print(f"\n Date: {record['date']}")
                print(f"    Check-in:  {record['check_in_time'].split()[1]}")
                print(f"    Check-out: {record['check_out_time'].split()[1]}")
                print(f"     Hours:     {record['hours_worked']} hours")
        print(f"\n{'='*70}")
        print(f" SUMMARY:")
        print(f"   Total Days Worked: {total_days}")
        print(f"   Total Hours: {round(total_hours, 2)} hours")
        if total_days > 0:
            print(f"   Average Hours/Day: {round(total_hours/total_days, 2)} hours")
        print(f"{'='*70}\n")
    def list_employees(self):
        """List all registered employees"""
        print(f"\n{'='*50}")
        print(f"REGISTERED EMPLOYEES ({len(self.employee_faces)})")
        print(f"{'='*50}")
        if not self.employee_faces:
            print("No employees registered yet.")
        else:
            for i, name in enumerate(self.employee_faces.keys(), 1):
                samples = len(self.employee_faces[name])
                print(f"{i}. {name} ({samples} face samples)")
        print(f"{'='*50}\n")
def main():
    print("\n" + "="*60)
    print("FACIAL RECOGNITION ATTENDANCE SYSTEM")
    print("="*60 + "\n")
    system = FaceAttendanceSystem()
    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Register New Employee")
        print("2. Check-In (Face Recognition)")
        print("3. Check-Out (Face Recognition)")
        print("4. View Employee Attendance Report")
        print("5. List All Employees")
        print("6. Exit")
        print("="*50)
        choice = input("\nEnter your choice (1-6): ").strip()
        if choice == '1':
            name = input("\nEnter employee name: ").strip()
            if name:
                system.register_employee(name)
            else:
                print("Invalid name!")
        elif choice == '2':
            system.recognize_face_from_camera(action='check_in')
        elif choice == '3':
            system.recognize_face_from_camera(action='check_out')
        elif choice == '4':
            name = input("\nEnter employee name: ").strip()
            if name:
                system.get_employee_report(name)
            else:
                print("Invalid name!")
        elif choice == '5':
            system.list_employees()
        elif choice == '6':
            print("\n✓ Thank you for using the Attendance System!")
            print("Goodbye!\n")
            break
        else:
            print("\n Invalid choice! Please enter a number between 1-6.")
if __name__ == "__main__":
    main()