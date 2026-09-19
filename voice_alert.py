import pyttsx3
import time
import threading
from queue import Queue, Full


# =========================
# VOICE ENGINE
# =========================

engine = pyttsx3.init()

# Speaking speed
engine.setProperty("rate", 145)

# Queue for voice alerts
speech_queue = Queue(maxsize=1)

last_alert = ""
last_alert_time = 0

# Repeat same alert after this many seconds
ALERT_INTERVAL = 5


# =========================
# SPEECH WORKER
# =========================

def speech_worker():
    while True:
        message = speech_queue.get()

        try:
            print("Speaking:", message)

            engine.say(message)
            engine.runAndWait()

        except Exception as e:
            print("Voice error:", e)

        finally:
            speech_queue.task_done()


# Start voice worker only once
voice_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

voice_thread.start()


# =========================
# SPEAK ALERT
# =========================

def speak_alert(message):
    global last_alert, last_alert_time

    current_time = time.time()

    # Don't repeat the exact same alert too quickly
    if (
        message == last_alert
        and current_time - last_alert_time < ALERT_INTERVAL
    ):
        print("Alert skipped:", message)
        return

    # Remember alert immediately
    last_alert = message
    last_alert_time = current_time

    # If another alert is waiting, replace it with the latest one
    try:
        speech_queue.put_nowait(message)

    except Full:
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except:
            pass

        try:
            speech_queue.put_nowait(message)
        except Full:
            pass


# =========================
# PROCESS ONE DETECTION
# =========================

def process_detection(detection):

    object_name = detection["name"]
    direction = detection["direction"]
    distance = detection["distance"]

    if distance <= 1:
        message = (
            f"Warning! {object_name} is extremely close "
            f"on the {direction}. Please stop."
        )

    elif distance <= 2:
        message = (
            f"Caution! {object_name} is very close "
            f"on the {direction}."
        )

    elif distance <= 5:
        message = (
            f"{object_name} detected on the {direction}, "
            f"about {distance} meters away."
        )

    else:
        message = (
            f"{object_name} detected on the {direction}."
        )

    speak_alert(message)


# =========================
# PROCESS MULTIPLE DETECTIONS
# =========================

def process_detections(detections):

    # Nearest object first
    detections.sort(key=lambda x: x["distance"])

    for detection in detections:
        process_detection(detection)
        