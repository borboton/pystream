from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# Abre la conexión con la cámara USB. El número 0 suele ser la cámara predeterminada.
cap = cv2.VideoCapture(2)

# Verifica si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

def generate_frames():
    while True:
        # Captura el fotograma de la cámara
        ret, frame = cap.read()

        # Si el fotograma se captura correctamente, conviértelo en formato JPEG
        if ret:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break

            # Convierte el búfer a bytes y envíalos como respuesta
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
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Inicia el servidor Flask en http://localhost:8080/
    app.run(host='0.0.0.0', port=8080)

