from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os


app = Flask(__name__)

capturas_dir = 'capturas'
os.makedirs(capturas_dir, exist_ok=True)

def guardar_captura(face_id, frame):
    nombre_archivo = f'captura_face_{face_id}.png'
    ruta_captura = os.path.join(capturas_dir, nombre_archivo)
    #print("type:", type(frame))
    cv2.imwrite(ruta_captura, frame)
    print(f'Captura guardada: {ruta_captura}')
    return face_id

def cargar_imagenes_entrenamiento():
    print("cargar_inagenes")
    imagenes_entrenamiento = []
    etiquetas_entrenamiento = []

    for etiqueta in os.listdir(capturas_dir):
        print(etiqueta)
        imagen_path = os.path.join(capturas_dir, etiqueta)
        print(imagen_path)
        try:
            imagen = face_recognition.load_image_file(imagen_path)
            encoding = face_recognition.face_encodings(imagen)[0]
            imagenes_entrenamiento.append(encoding)
            etiquetas_entrenamiento.append(etiqueta)
        except Exception as e:
            print(str(e))

    print(imagenes_entrenamiento)
    return imagenes_entrenamiento, etiquetas_entrenamiento

def entrenar_modelo():
    imagenes_entrenamiento, etiquetas_entrenamiento = cargar_imagenes_entrenamiento()
    return imagenes_entrenamiento, etiquetas_entrenamiento

def reconocer_rostro(frame, modelo, etiquetas):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_id = f"{top}{right}{bottom}{left}"

        
           #guardar_captura(face_id, np.array(frame)) 
        print(type(modelo))
        matches = face_recognition.compare_faces(modelo, face_encoding)
        nombre = "Desconocido"

        if True in matches:
            print("match:" , matches)
            first_match_index = matches.index(True)
            print("first_match_index:", first_match_index)
            nombre = f"{etiquetas[first_match_index]}"
            print("nombre:", nombre)
        #else:
        #    guardar_captura(nombre, frame)
            

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, nombre, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def generate_frames(modelo, etiquetas):
    while True:
        ret, frame = cap.read()

        if ret:
            #print(modelo, etiquetas)
            reconocer_rostro(frame, modelo, etiquetas)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para la transmisión de video
@app.route('/video')
def video():
    return Response(generate_frames(modelo, etiquetas), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    face_appeared = {}
    modelo, etiquetas = entrenar_modelo()
    cap = cv2.VideoCapture(2)
    app.run(host='0.0.0.0', port=8080)
