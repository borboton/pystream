from flask import Flask, render_template, Response, send_file, request
from flask_socketio import SocketIO
import time
import cv2
import os
import hashlib
import logging

class FaceRecognitionApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.capturas_dir = 'capturas'
        self.imagenes_entrenamiento_dir = 'entrenamiento'
        self.default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        os.makedirs(self.capturas_dir, exist_ok=True)
        os.makedirs(self.imagenes_entrenamiento_dir, exist_ok=True)

        self.face_appeared = {}
        self.modelo, self.etiquetas = self.entrenar_modelo()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)

        @self.socketio.on('connect')
        def handle_connect():
            logging.debug("Cliente conectado: %s", request.sid)
            image_files = [f for f in os.listdir(self.capturas_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            image_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.capturas_dir, x)), reverse=True)
            self.socketio.emit('image_list', {'images': image_files}, room=request.sid)

        @self.socketio.on('disconnect')
        def handle_disconnect():
            logging.info("Cliente desconectado: %s", request.sid)

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/video')
        def video():
            return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.app.route('/getface/<filename>')
        def get_face(filename):
            path = os.path.abspath(self.capturas_dir)
            ruta_captura = os.path.join(path, filename)

            if os.path.isfile(ruta_captura):
                time.sleep(0.5)
                return send_file(ruta_captura, mimetype='image/png', as_attachment=True)
            else:
                return "Archivo no encontrado", 404

    def run(self):
        logging.basicConfig(format=('%(asctime)s | %(threadName)s - %(message)s'), level=logging.DEBUG)
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        logging.info("Start..")
        self.socketio.run(self.app, host='0.0.0.0', port=8080, allow_unsafe_werkzeug=True)

    def entrenar_modelo(self):
        # Implementation of your training logic...
        imagenes_entrenamiento = []
        etiquetas_entrenamiento = []

        for etiqueta in os.listdir(self.imagenes_entrenamiento_dir):
            imagen_path = os.path.join(self.imagenes_entrenamiento_dir, etiqueta)
            try:
                imagen = cv2.imread(imagen_path)
                encoding = face_recognition.face_encodings(imagen)[0]
                imagenes_entrenamiento.append(encoding)
                etiquetas_entrenamiento.append(etiqueta)
            except Exception as e:
                print(str(e))

        return imagenes_entrenamiento, etiquetas_entrenamiento

    def guardar_captura(self, face_id, frame, location):
        nombre_archivo = f'{face_id}'
        ruta_captura = os.path.join(self.capturas_dir, nombre_archivo)
        top, right, bottom, left = location
        cara_recortada = frame[top:bottom, left:right]
        cv2.imwrite(ruta_captura, cara_recortada)

        logging.info("Captura guardada: %s", ruta_captura)
        return face_id

    def reconocer_rostro(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_id = f"{top}{right}{bottom}{left}"
            matches = face_recognition.compare_faces(self.modelo, face_encoding)
            nombre = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                nombre = f"{self.etiquetas[first_match_index]}"
            else:
                capface = f"Recognized-{face_id}.png"
                location = (top, right, bottom, left)
                self.guardar_captura(capface, frame, location)
                self.socketio.emit('actualizacion', {'nombre': capface})
                self.etiquetas.append(capface)

            if nombre not in self.face_appeared:
                self.face_appeared[nombre] = 1
            else:
                self.face_appeared[nombre] += 1

            cv2.rectangle(frame, (left, top), (right, bottom), (1, 254, 4), 2)
            cv2.putText(frame, "Recognized", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 254, 4), 2)

    def face_detector(self, frame):
        face_locations = face_recognition.face_locations(frame)
        for face_location in face_locations:
            top, right, bottom, left = face_location
            nombre = "Desconocido"
            face_id = f"{top}{right}{bottom}{left}"
            name = f"{nombre}_{face_id}"
            capface = f"captura_face_{name}.png"
            cv2.rectangle(frame, (left, top), (right, bottom), (1, 254, 4), 2)
            self.guardar_captura(name, frame, face_location)
            self.socketio.emit('actualizacion', {'nombre': capface})

    def obj_detector(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        obj = self.default.detectMultiScale(rgb_frame,
                                       scaleFactor=(2.05),
                                       minNeighbors=4,
                                       minSize=(40, 40),
                                       maxSize=(80, 80))

        for (left, top, right, bottom) in obj:
            location = (top, right, bottom, left)
            face_id = f"{top}{right}{bottom}{left}"
            cv2.rectangle(frame, (left, top), (left + right, top + bottom), (0, 0, 255), 1)
            capface = f"Recognized-{face_id}.png"
            self.socketio.emit('actualizacion', {'nombre': capface})
            body_roi = frame[top:top + bottom, left:left + right]
            filename = os.path.abspath(f"./capturas/{capface}")
            cv2.imwrite(filename, body_roi)
            logging.debug("imwrite: %s", filename)

            if face_id not in self.face_appeared:
                self.face_appeared[face_id] = 1
            else:
                self.face_appeared[face_id] += 1

    def generate_frames(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.obj_detector(frame)

                ret, buffer = cv2.imencode('.jpg', frame)

                if not ret:
                    break

                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == '__main__':
    app_instance = FaceRecognitionApp()
    app_instance.run()

