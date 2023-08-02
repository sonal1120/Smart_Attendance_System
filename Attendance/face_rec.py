import face_recognition
import pickle
import datetime
import redis
import numpy as np

class FaceRecognition:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.known_roles = []
        self.redis_db = redis.Redis(host='18681.c276.us-east-1-2.ec2.cloud.redislabs.com', port=18681, db=0)

    def load_data(self):
        try:
            with open('encodings.pickle', 'rb') as f:
                data = pickle.load(f)
                self.known_encodings = data['encodings']
                self.known_names = data['names']
                self.known_roles = data['roles']
            return True
        except FileNotFoundError:
            return False

    def save_data(self):
        data = {
            'encodings': self.known_encodings,
            'names': self.known_names,
            'roles': self.known_roles
        }
        with open('encodings.pickle', 'wb') as f:
            pickle.dump(data, f)

    def register_person(self, name, role, image):
        encoding = face_recognition.face_encodings(image)[0]
        self.known_encodings.append(encoding)
        self.known_names.append(name)
        self.known_roles.append(role)
        self.save_data()

    def recognize_faces(self, image):
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        face_names = []
        face_roles = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            name = 'Unknown'
            role = 'Unknown'

            if True in matches:
                matched_indices = [i for i, match in enumerate(matches) if match]
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.6:
                    name = self.known_names[best_match_index]
                    role = self.known_roles[best_match_index]

            face_names.append(name)
            face_roles.append(role)

        return face_names, face_roles

    def save_log(self, name, role):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {'timestamp': timestamp, 'name': name, 'role': role}
        self.redis_db.rpush('attendance_logs', log_entry)

    def retrieve_logs(self):
        logs = self.redis_db.lrange('attendance_logs', 0, -1)
        log_entries = []
        for log in logs:
            log_entry = eval(log.decode('utf-8'))
            log_entries.append(log_entry)
        return log_entries
