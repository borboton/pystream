from flask import Flask, render_template, Response
from flask import send_file, request
from flask_socketio import SocketIO, join_room
import time
import cv2
#import face_recognition
import numpy as np
import os
import glob
import hashlib
import logging



#default = cv2.HOGDescriptor()
#default.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
#default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml')
#default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#default = cv2.HOGDescriptor()


app = Flask(__name__)
socketio = SocketIO(app)

capturas_dir = 'capturas'
imagenes_entrenamiento_dir = 'entrenamiento'

os.makedirs(capturas_dir, exist_ok=True)
os.makedirs(imagenes_entrenamiento_dir, exist_ok=True)


def guardar_captura(face_id, frame, location):
    nombre_archivo = f'{face_id}'
    ruta_captura = os.path.join(capturas_dir, nombre_archivo)    
    top, right, bottom, left = location
    cara_recortada = frame[top:bottom, left:right]
    cv2.imwrite(ruta_captura, cara_recortada)

    logging.info("Captura guardada: %s", ruta_captura)
    return face_id

def cargar_imagenes_entrenamiento():
    logging.debug("Cargarndo inagenes")
    imagenes_entrenamiento = []
    etiquetas_entrenamiento = []

    for etiqueta in os.listdir(imagenes_entrenamiento_dir):
     
        imagen_path = os.path.join(imagenes_entrenamiento_dir, etiqueta)
        try:
            imagen = face_recognition.load_image_file(imagen_path)
            encoding = face_recognition.face_encodings(imagen)[0]
            imagenes_entrenamiento.append(encoding)
            etiquetas_entrenamiento.append(etiqueta)
        except Exception as e:
            print(str(e))

    return imagenes_entrenamiento, etiquetas_entrenamiento

def entrenar_modelo():
    imagenes_entrenamiento, etiquetas_entrenamiento = cargar_imagenes_entrenamiento()
    logging.info("Imagenes_entrenamiento: %d", len(imagenes_entrenamiento))
    return imagenes_entrenamiento, etiquetas_entrenamiento

def reconocer_rostro(frame, modelo, etiquetas):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    '''
    if face_encodings:
        sha256_hash = hashlib.sha256(face_encodings[0].tobytes())
        if sha256_hash not in face_appeared:
            face_appeared[sha256_hash] = 1
        else:
            face_appeared[sha256_hash] = +1
    '''
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_id = f"{top}{right}{bottom}{left}"
        matches = face_recognition.compare_faces(modelo, face_encoding)
        nombre = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            nombre = f"{etiquetas[first_match_index]}"
        else:
            capface = f"Recognized-{face_id}.png"
            location = (top, right, bottom, left)
            guardar_captura(capface, frame, location)
            socketio.emit('actualizacion', {'nombre': capface})
            #modelo.append(face_encodings[0])
            etiquetas.append(capface)

        if nombre not in face_appeared:
            face_appeared[nombre] = 1
        else:
            face_appeared[nombre] += 1

        cv2.rectangle(frame, (left, top), (right, bottom), (1, 254, 4), 2)
        cv2.putText(frame, "Recognized", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 254, 4), 2)
    
    return face_appeared
    

def face_detector(frame):
    face_locations = face_recognition.face_locations(frame)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        nombre = "Desconocido"
        face_id = f"{top}{right}{bottom}{left}"
        name = f"{nombre}_{face_id}"
        capface = f"captura_face_{name}.png"
        cv2.rectangle(frame, (left, top), (right, bottom), (1, 254, 4), 2)
        guardar_captura(name, frame, face_location)
        socketio.emit('actualizacion', {'nombre': capface})


@socketio.on('connect')
def handle_connect():
    logging.debug("Cliente conectado: %s", request.sid)
    image_files = [f for f in os.listdir(capturas_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    #img = sorted(image_files, key=lambda x: (int(x[0].split("-")[0]) if x[1][:1].isdigit() else 999, x))
    #files = list(filter(os.path.isfile, glob.glob(capturas_dir + "*")))
    image_files.sort(key=lambda x: os.path.getmtime(f"./capturas/{x}"), reverse=True)
    socketio.emit('image_list', { 'images': image_files }, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    logging.info("Cliente desconectado: %s", request.sid)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(modelo, etiquetas), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/getface/<filename>')
def get_face(filename):
    path = os.path.abspath(f"{capturas_dir}")
    ruta_captura = os.path.join(path, f'{filename}')

    if os.path.isfile(ruta_captura):
        time.sleep(0.5)
        return send_file(ruta_captura, mimetype='image/png', as_attachment=True)
    else:
        return "Archivo no encontrado", 404

def obj_detector(frame, modelo, etiquetas):
    #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face
    #obj = default.detectMultiScale(rgb_frame, scaleFactor=1.1, minSize=(30, 30))

    # body
    #obj = default.detectMultiScale(rgb_frame, 
                                   #scaleFactor=(1.1), 
                                   #minNeighbors=5,
                                   #minSize=(40,10))

    obj = default.detectMultiScale(rgb_frame, 
                                   scaleFactor=(2.05),
                                   minNeighbors=4,
                                   minSize=(40,40),
                                   maxSize=(80,80))

    # face
    #obj = default.detectMultiScale(rgb_frame, 
                                   #scaleFactor=(2.05),
                                   #minNeighbors=4,
                                   #minSize=(12,12),
                                   #maxSize=(100,100))

    for (left,top,right,bottom) in obj:
        location  = ( top, right, bottom, left )
        face_id = f"{top}{right}{bottom}{left}"
        cv2.rectangle(frame, (left,top),(left+right,top+bottom),(0,0,255),1)
        capface = f"Recognized-{face_id}.png"
        socketio.emit('actualizacion', {'nombre': capface})
        body_roi = frame[top:top+bottom, left:left+right]
        filename = os.path.abspath(f"./capturas/{capface}")
        cv2.imwrite(filename, body_roi)
        logging.debug("imwrite: %s", filename)

        if face_id not in face_appeared:
            face_appeared[face_id] = 1
        else:
            face_appeared[face_id] += 1

def generate_frames(modelo, etiquetas):
    while True:
        ret, frame = cap.read()
        if ret:
            #print(modelo, etiquetas)
            #reconocer_rostro(frame, modelo, etiquetas)
            #face_detector(frame)
            obj_detector(frame, modelo, etiquetas)

            ret, buffer = cv2.imencode('.jpg', frame)

            if not ret:
                break

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


if __name__ == '__main__':
    logging.basicConfig(format=('%(asctime)s | %(threadName)s - %(message)s'), level=logging.DEBUG)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    logging.info("Start..")
    face_appeared = {}
    modelo, etiquetas = entrenar_modelo()
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    #app.run(host='0.0.0.0', port=8080)
    socketio.run(app, host='0.0.0.0', port=8080, allow_unsafe_werkzeug=True)

