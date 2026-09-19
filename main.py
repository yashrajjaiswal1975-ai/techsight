import cv2
from ultralytics import YOLO
from voice_alert import speak_alert
import threading
import queue
import time


# =================================================
# YOLO SETUP
# =================================================

print("Loading YOLO...")

model = YOLO("yolo11n.pt")



# =================================================
# CAMERA SETUP
# =================================================

print("\n================================")
print("       AI VISION ASSISTANT")
print("================================")
print("1. Mac Camera")
print("2. Samsung Phone Camera (Camo)")
print("3. Exit")
print("================================")

choice = input("Enter your choice: ")


# -------------------------------------------------
# MAC CAMERA
# -------------------------------------------------

if choice == "1":

    print("Opening Mac camera...")

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION
    )


# -------------------------------------------------
# SAMSUNG CAMERA THROUGH CAMO
# -------------------------------------------------

elif choice == "2":

    print("Opening Samsung camera through Camo...")

    cap = cv2.VideoCapture(
        1,
        cv2.CAP_AVFOUNDATION
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)


# -------------------------------------------------
# EXIT
# -------------------------------------------------

elif choice == "3":

    print("Exiting...")
    exit()


# -------------------------------------------------
# INVALID CHOICE
# -------------------------------------------------

else:

    print("Invalid choice.")
    exit()


# -------------------------------------------------
# CHECK CAMERA
# -------------------------------------------------

if not cap.isOpened():

    print("ERROR: Could not open selected camera.")

    if choice == "2":
        print("Make sure Camo Studio is running")
        print("and your Samsung phone is connected.")

    exit()


print("Camera opened successfully!")
print("AI Vision Assistant started. Press Q to quit.")

# =================================================
# VOICE QUEUE
# =================================================

# Multiple objects ko queue karne ke liye
voice_queue = queue.Queue(maxsize=5)


def voice_worker():

    while True:

        message = voice_queue.get()

        if message is None:
            voice_queue.task_done()
            break

        try:
            print("VOICE:", message)

            speak_alert(message)

        except Exception as e:
            print("Voice error:", e)

        finally:
            voice_queue.task_done()


voice_thread = threading.Thread(
    target=voice_worker,
    daemon=True
)

voice_thread.start()


# =================================================
# DIRECTION
# =================================================

def get_direction(x_center, frame_width):

    if x_center < frame_width / 3:
        return "left"

    elif x_center < (2 * frame_width / 3):
        return "center"

    else:
        return "right"


# =================================================
# DISTANCE
# =================================================

def get_distance(box_height, frame_height):

    ratio = box_height / frame_height

    if ratio > 0.55:
        return 0.8

    elif ratio > 0.30:
        return 1.5

    elif ratio > 0.15:
        return 3.0

    else:
        return 6.0
 # =================================================
# INFORMATION PANEL
# =================================================

def draw_info_panel(frame, detections):

    # Background panel
    cv2.rectangle(
        frame,
        (10, 10),
        (350, 120 + min(len(detections), 5) * 30),
        (0, 0, 0),
        -1
    )

    # Title
    cv2.putText(
        frame,
        "AI VISION ASSISTANT",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    y = 65

    # System status
    cv2.putText(
        frame,
        "SYSTEM: RUNNING",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    y += 25

    # Voice status
    cv2.putText(
        frame,
        "VOICE: ON",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    y += 25

    # Object count
    cv2.putText(
        frame,
        f"OBJECTS: {len(detections)}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    y += 30

    # Object details
    for detection in detections[:5]:

        text = (
            f"{detection['name']} | "
            f"{detection['distance']}m | "
            f"{detection['direction']}"
        )

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        y += 30


# =================================================
# MAIN VARIABLES
# =================================================

frame_count = 0

last_results = None

last_detections = []
 
last_annotated_frame = None

last_voice_time = 0

VOICE_INTERVAL = 2.0

# =================================================
# OBJECT ALERT MEMORY
# =================================================

object_alert_state = {}


# =================================================
# MAIN CAMERA LOOP
# =================================================

while True:

    # -------------------------------------------------
    # READ CAMERA FRAME
    # -------------------------------------------------

    ret, frame = cap.read()

    if not ret:

        print("WARNING: Camera frame failed")

        time.sleep(0.05)

        continue


        frame_height, frame_width = frame.shape[:2]
        frame = cv2.resize(frame, (640, 480))
    frame_height, frame_width = frame.shape[:2]

    frame_count += 1


    # -------------------------------------------------
    # YOLO ONLY EVERY 3RD FRAME
    # -------------------------------------------------

    if frame_count % 2 == 0:
        results = model(
            frame,
            imgsz=320,
            conf=0.45,
            max_det=20,
            verbose=False,
            device="mps"
        )

        last_results = results

        last_annotated_frame = results[0].plot()

        detections = []

        boxes = results[0].boxes


        # -------------------------------------------------
        # PROCESS DETECTED OBJECTS
        # -------------------------------------------------

        for box in boxes[:20]:

            confidence = float(box.conf[0])

            if confidence < 0.40:
                continue


            class_id = int(box.cls[0])

            object_name = model.names[class_id]


            # Bounding box

            x1, y1, x2, y2 = box.xyxy[0].tolist()


            # Center

            x_center = (x1 + x2) / 2


            # Height

            box_height = y2 - y1


            # Direction

            direction = get_direction(
                x_center,
                frame_width
            )


            # Approximate distance

            distance = get_distance(
                box_height,
                frame_height
            )


            # Detection information

            detection = {

                "name": object_name,

                "direction": direction,

                "distance": distance,

                "confidence": round(confidence, 2),

                "x_center": x_center

            }


            detections.append(detection)


        # -------------------------------------------------
        # VOICE LOGIC
        # -------------------------------------------------

        last_detections = detections.copy()
        current_time = time.time()


        if detections and current_time - last_voice_time >= VOICE_INTERVAL:

            # Closest object first

            sorted_detections = sorted(
                detections,
                key=lambda x: x["distance"]
            )


            messages_to_speak = []


            # Objects currently visible

            visible_keys = set()


            for detection in sorted_detections[:2]:

                object_name = detection["name"]

                direction = detection["direction"]

                current_x = detection["x_center"]

                current_distance = detection["distance"]


                # Object identity

                object_key = f"{object_name}_{direction}"


                visible_keys.add(object_key)


                # -------------------------------------------------
                # CREATE STATE FOR NEW OBJECT
                # -------------------------------------------------

                if object_key not in object_alert_state:

                    object_alert_state[object_key] = {

                        "count": 0,

                        "x": current_x,

                        "distance": current_distance,

                        "missing": 0

                    }


                state = object_alert_state[object_key]


                # Object is visible again

                state["missing"] = 0


                # -------------------------------------------------
                # MOVEMENT DETECTION
                # -------------------------------------------------

                moved = False


                if state["x"] is not None:

                    # Horizontal movement

                    if abs(current_x - state["x"]) > 40:

                        moved = True


                    # Distance category changed

                    if current_distance != state["distance"]:

                        moved = True


                # -------------------------------------------------
                # ALERT DECISION
                # -------------------------------------------------

                should_alert = False


                # First 2 alerts

                if state["count"] < 2:

                    should_alert = True


                # After 2 alerts, only movement triggers alert

                elif moved:

                    should_alert = True

                    # Restart 2-alert cycle

                    state["count"] = 0


                # -------------------------------------------------
                # CREATE VOICE MESSAGE
                # -------------------------------------------------

                if should_alert:

                    if current_distance <= 1:

                        message = (

                            f"Warning! {object_name} is extremely close "

                            f"on the {direction}. Please stop."

                        )


                    elif current_distance <= 2:

                        message = (

                            f"Caution! {object_name} is very close "

                            f"on the {direction}."

                        )


                    elif current_distance <= 5:

                        message = (

                            f"{object_name} detected on the {direction}, "

                            f"about {current_distance} meters away."

                        )


                    else:

                        message = (

                            f"{object_name} detected on the {direction}."

                        )


                    messages_to_speak.append(message)


                    state["count"] += 1


                # -------------------------------------------------
                # SAVE CURRENT POSITION
                # -------------------------------------------------

                state["x"] = current_x

                state["distance"] = current_distance


            # -------------------------------------------------
            # RESET OBJECTS THAT DISAPPEAR
            # -------------------------------------------------

            for key in list(object_alert_state.keys()):

                if key not in visible_keys:

                    object_alert_state[key]["missing"] += 1


                    # If object disappears for several detections,
                    # treat it as a new object next time

                    if object_alert_state[key]["missing"] >= 5:

                        del object_alert_state[key]


            # -------------------------------------------------
            # ADD ALL VOICE MESSAGES TO QUEUE
            # -------------------------------------------------
            if messages_to_speak:

                try:

                    # Speak only the most important alert
                    voice_queue.put_nowait(messages_to_speak[0])

                except queue.Full:

                    print("Voice queue busy - skipping alert")

                last_voice_time = current_time


        # -------------------------------------------------
        # PRINT DETECTIONS
        # -------------------------------------------------

        #if detections:

        #    print("\nDetected objects:")

        #    for detection in detections:

        #        print(detection)

            # =================================================
    # DISPLAY
    # =================================================

    # Display happens EVERY frame,
    # not only when voice is triggered.

    if last_annotated_frame is not None:

        annotated_frame = last_annotated_frame.copy()

    else:

        annotated_frame = frame.copy()

    # Information panel

    draw_info_panel(
        annotated_frame,
        last_detections
    )


    cv2.imshow(
        "AI Vision Assistant - Object Detection",
        annotated_frame
    )


    # =================================================
    # QUIT
    # =================================================
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        print("ESC pressed - stopping program.")
        break
        


# =================================================
# CLEANUP
# =================================================

cap.release()

voice_queue.put(None)

cv2.destroyAllWindows()

print("AI Vision Assistant stopped.")