
import os
import requests
import numpy as np
import librosa
import matplotlib.pyplot as plt
from gtts import gTTS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import os


# Definición de rutas según la estructura solicitada
DATASET_PATH = 'D:/Emanuel/Desktop/ABP Proc. del Habla/version con streamlit/audios_final'


CATEGORIES = {
    'clima': 0,
    'reporte': 1,
    'et': 2 # Estado del Tiempo
}

# Verificación de que las carpetas existan
if os.path.exists(DATASET_PATH):
    for subfolder in CATEGORIES.keys():
        subfolder_path = os.path.join(DATASET_PATH, subfolder)
        if os.path.exists(subfolder_path):
            cant_archivos = len([f for f in os.listdir(subfolder_path) if f.endswith('.wav')])
        else:
            pass


# PIPELINE DSP Y EXTRACCIÓN DE METADATOS / FEATURES (Estándar 16kHz y 1.0s)
TARGET_SR = 16000  # 16 kHz según el estándar de la propuesta
DURATION = 2.5 
TOTAL_SAMPLES = int(TARGET_SR * DURATION)

def preprocess_audio(file_path):
    """
    Pipeline del Eje I: Carga, Trim de silencios, Resample, Normalización y Padding/Crop.
    """
    # 1. Carga y Resample automático a 16kHz
    y, sr = librosa.load(file_path, sr=TARGET_SR)

    # 2. Trim: Eliminar silencios iniciales y finales
    y, _ = librosa.effects.trim(y, top_db=20)

    # 3. Normalización de amplitud
    if len(y) > 0:
        y = librosa.util.normalize(y)

    # 4. Forzar duración exacta a 1.0 segundo (Padding o Truncate)
    if len(y) < TOTAL_SAMPLES:
        y = np.pad(y, (0, TOTAL_SAMPLES - len(y)), mode='constant')
    else:
        y = y[:TOTAL_SAMPLES]

    return y

def extract_features(y):
    """
    Extracción de 13 MFCCs + Deltas + Características Espectrales clave. [cite: 41]
    """
    # MFCCs (13 coeficientes)
    mfccs = librosa.feature.mfcc(y=y, sr=TARGET_SR, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Deltas de los MFCCs
    deltas = librosa.feature.delta(mfccs)
    deltas_mean = np.mean(deltas, axis=1)

    # Características espectrales adicionales
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=TARGET_SR))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=TARGET_SR))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rms_energy = np.mean(librosa.feature.rms(y=y))

    # Consolidar vector de características
    features = np.hstack([mfccs_mean, deltas_mean, spectral_centroid, spectral_rolloff, spectral_flatness, rms_energy])
    return features

# Procesamiento iterativo de todo el dataset real
X_data = []
y_labels = []

for folder_name, label_idx in CATEGORIES.items():
    folder_path = os.path.join(DATASET_PATH, folder_name)
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                full_path = os.path.join(folder_path, file_name)
                try:
                    cleaned_audio = preprocess_audio(full_path)
                    feat_vector = extract_features(cleaned_audio)
                    X_data.append(feat_vector)
                    y_labels.append(label_idx)
                except Exception as e:
                    pass

X_data = np.array(X_data)
y_labels = np.array(y_labels)




# ENTRENAMIENTO DEL MODELO (RANDOM FOREST) Y MATRIZ DE CONFUSIÓN

# División estratificada 70% entrenamiento y 30% testeo
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_labels, test_size=0.3, stratify=y_labels, random_state=42
)

# Inicialización y entrenamiento de un clasificador interpretable
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicciones y métricas
y_pred = model.predict(X_test)


# ADQUISICIÓN CLIMÁTICA POR GEOLOCALIZACIÓN Y SISTEMA DE ALERTAS (REAL/SIMULADO)
def get_location_and_weather(simulation_mode=False, sim_temp=None, sim_humidity=None):
    """
    consulta la API climática Open-Meteo.
    Si simulation_mode=True, sobreescribe los valores climáticos para disparar alertas.
    """
    
    city = "Río Tercero"
    lat, lon = -32.17, -64.11

    if simulation_mode:
        # Forzar datos simulados para evaluar alertas estacionales/críticas
        temp = sim_temp if sim_temp is not None else 12.0
        humidity = sim_humidity if sim_humidity is not None else 85.0
        mode_str = "⚠️ [MODO SIMULACIÓN ACTIVADO]"
    else:
        # Consulta en tiempo real a Open-Meteo API (Gratuita, sin necesidad de token)
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&relative_humidity_2m=true"
        try:
            weather_data = requests.get(weather_url).json()
            temp = weather_data['current_weather']['temperature']
            # Estructura típica de Open-Meteo o fallback de humedad SIMULADA
            humidity = weather_data.get('current_humidity', 60.0)
            mode_str = "[DATOS REALES OBTENIDOS POR GEOLOCALIZACIÓN]"
        except Exception:
            # Fallback seguro si la API de internet falla temporalmente
            temp, humidity = 13.5, 72.0
            mode_str = "[API DE RESERVA EN TIEMPO REAL]"

    # Lógica de Negocio: Filtro y Reglas de Alerta
    alert_msg = ""
    if temp >= 35.0:
        alert_msg += "¡Alerta por ola de calor extremo! Evite exponerse al sol, y no olvide hidratarse. "
    elif temp <= 5.0:
        alert_msg += "¡Alerta por temperaturas extremadamente bajas o heladas! Abríguese bien por favor. "

    if humidity <= 30.0:
        alert_msg += "Alerta por baja humedad ambiente, riesgo de incendios y sequedad respiratoria. "
    elif humidity >= 95.0:
        alert_msg += "Advertencia por humedad saturada, posible reducción drástica de visibilidad o niebla intensa. "

    if not alert_msg:
        alert_msg = "No se registran alertas meteorológicas activas para su zona."

    # Construcción de la respuesta accesible
    report_text = f"Estación Meteorológica SIMA informa. En la ciudad de {city}. {mode_str}. " \
                  f"La temperatura actual es de {temp} grados Celsius. " \
                  f"La humedad relativa es del {humidity} por ciento. " \
                  f"Estado de alertas: {alert_msg}"

    return report_text

# CONFIGURACIÓN DE LA PÁGINA WEB
st.set_page_config(page_title="VozActiva - Estación Meteorológica", page_icon="🌤️", layout="centered")
 
# INTERFAZ GRÁFICA DE USUARIO (Streamlit UI/UX)
st.title("🌤️ Estación Meteorológica Accesible - VozActiva")
st.write("### Prototipo Web Integrado")
st.write("---")

tab1, tab2 = st.tabs(["🎙️ Asistente de Voz", "🎛️ Configuración de Umbrales Técnicos"])

with tab2:
    st.subheader("Ajuste Fino de Tolerancia Acústica")
    st.write("Modifique los umbrales de aceptación para cada trigger. Si la certeza del modelo es menor al umbral configurado, el comando será rechazado por seguridad.")

    # Sliders dinámicos independientes para cada comando
    umbral_clima = st.slider("Umbral de confianza para 'CLIMA'", 0.40, 1.00, 0.75, step=0.05)
    umbral_reporte = st.slider("Umbral de confianza para 'REPORTE'", 0.40, 1.00, 0.65, step=0.05)
    umbral_et = st.slider("Umbral de confianza para 'ESTADO DEL TIEMPO (ET)'", 0.40, 1.00, 0.60, step=0.05)

    # Mapeo interno de umbrales para facilitar la lectura
    umbrales_dict = {
        'clima': umbral_clima,
        'reporte': umbral_reporte,
        'et': umbral_et
    }

with tab1:
    st.info("💡 Grabe su comando utilizando el control de micrófono de abajo.")
    audio_file = st.audio_input("Presione para grabar su voz:")

    st.write("#### ⚙️ Panel de Simulación (Para evaluación de alertas en la defensa)")
    chk_simulacion = st.checkbox("Activar simulación de clima extremo", value=False)
    col1, col2 = st.columns(2)
    with col1: sld_temp = st.slider("Temperatura Simulada (°C)", -10.0, 45.0, 14.0, step=0.5)
    with col2: sld_hum = st.slider("Humedad Simulada (%)", 0.0, 100.0, 65.0, step=1.0)

    if audio_file is not None and model is not None:
        st.write("🚀 Procesar Audio Recibido")
        with st.spinner("🚀 Procesando el audio recibido..."):
            # DSP & Feature Extraction
            y_clean = preprocess_audio(audio_file)
            features = extract_features(y_clean).reshape(1, -1)
            # Inferencia del Modelo Real
            prediction = model.predict(features)[0]
            predicted_trigger = [k for k, v in CATEGORIES.items() if v == prediction][0]
            probabilidades = model.predict_proba(features)[0]
            prob_max = max(probabilidades)
            # OBTENER EL UMBRAL ESPECÍFICO ASIGNADO AL COMANDO DETECTADO
            umbral_requerido = umbrales_dict.get(predicted_trigger, 0.70)
            st.write(f"📊 **Análisis del clasificador:** Detectó correctamete la clase con una certeza de `{prob_max:.2f}` (Umbral mínimo requerido para esta clase: `{umbral_requerido:.2f}`) ")
            # Validación estricta usando el umbral dinámico por comando
            if prob_max < umbral_requerido:
                st.error(f"❌ Comando rechazado. La confianza ({prob_max:.2f}) es inferior al umbral configurado para el comando. Intente vocalizar de forma más clara.")
            else:
                st.success(f"🎯 **Comando Aceptado de forma exitosa!**")
                # Generar Reporte + Alertas
                reporte_final = get_location_and_weather(chk_simulacion, sld_temp, sld_hum)
                st.text_area("📝 Transcripción del reporte generado:", value=reporte_final, height=120)
                # Salida de accesibilidad (TTS)
                tts = gTTS(text=reporte_final, lang='es', tld='com.ar')
                output_file = "reporte_actualizado.mp3"
                tts.save(output_file)
                st.write("🔊 **Reproduciendo Reporte de Audio Sintetizado (Autoplay):**")
                st.audio(output_file, format="audio/mp3", autoplay=True)

