"""
ROBOT NAV v9 — Collecte de données scientifiques
=================================================
Objectif : comparer 4 méthodes de localisation pour rapport scolaire
  - Odometrie seule (encodeurs)
  - Ultrasons (distances + correction cap)
  - Vision (ArUco markers)
  - Fusion (EKF : odom + IMU + ArUco)

Export direct au format robot_data.json compatible avec le script analyse Python.
"""
import io, json, time, math, threading, uvicorn
import numpy as np
import os, serial

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from picamera2 import Picamera2
    import cv2
    import cv2.aruco as aruco
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

from PIL import Image

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ═══════════════════════════════════════════════════
# CONSTANTES MECANIQUES
# 7PPR x 150 reducteur x4 quadrature = 4200 ticks/tour
# Roue 44mm → 0.0329mm/tick
# ═══════════════════════════════════════════════════
TICKS_PER_REV = 4200
WHEEL_DIAM    = 0.044
WHEEL_C       = math.pi * WHEEL_DIAM
M_PER_TICK    = WHEEL_C / TICKS_PER_REV
WHEEL_BASE    = 0.17

def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

# ═══════════════════════════════════════════════════
# BASE DE DONNEES EXPERIENCE
# Format compatible avec robot_data.json du script Python
# ═══════════════════════════════════════════════════
DATA_FILE = "robot_data.json"

def load_exp_data():
    if os.path.exists(DATA_FILE):
        try:
            return json.load(open(DATA_FILE))
        except:
            pass
    return {"methods": {}, "time_series": {}, "trajectories": {}}

def save_exp_data(data):
    json.dump(data, open(DATA_FILE, "w"), indent=2)

exp_data     = load_exp_data()
exp_lock     = threading.Lock()

# Session d'expérience courante
session = {
    "running":     False,
    "mode":        "position",   # "position" | "time" | "trajectory"
    "start_time":  0,
    "step":        0,
    "target_x":    100.0,  # cm — position cible
    "target_y":    0.0,
    "log":         [],     # log temps réel
    "auto_stop_distance": 50,  # cm — s'arrête tous les Xcm
    "last_stop_dist": 0,
}

# ═══════════════════════════════════════════════════
# PICO STATE
# ═══════════════════════════════════════════════════
pico_state = {
    "connected": False,
    "dist_c": 999, "dist_l": 999, "dist_r": 999,
    "ticks_l": 0, "ticks_r": 0,
    "yaw": 0.0, "imu_ok": False,
    "last_update": 0, "nav_state": "STOP"
}
pico_lock = threading.Lock()

# ═══════════════════════════════════════════════════
# METHODE 1 : ODOMETRIE PURE
# ═══════════════════════════════════════════════════
odom = {"x": 0.0, "y": 0.0, "theta": 0.0, "active": False}
odom_lock = threading.Lock()
_odom_tl_prev = 0
_odom_tr_prev = 0

def odom_reset(x=0.0, y=0.0, theta=0.0):
    global _odom_tl_prev, _odom_tr_prev
    with odom_lock:
        odom["x"] = x; odom["y"] = y; odom["theta"] = theta; odom["active"] = True
    with pico_lock:
        _odom_tl_prev = pico_state["ticks_l"]
        _odom_tr_prev = pico_state["ticks_r"]

def odom_update():
    global _odom_tl_prev, _odom_tr_prev
    with pico_lock:
        tl = pico_state["ticks_l"]
        tr = pico_state["ticks_r"]
    dtl = tl - _odom_tl_prev
    dtr = tr - _odom_tr_prev
    _odom_tl_prev = tl
    _odom_tr_prev = tr
    if dtl == 0 and dtr == 0:
        return
    dl   = dtl * M_PER_TICK
    dr   = dtr * M_PER_TICK
    dist = (dl + dr) / 2.0
    dth  = (dr - dl) / WHEEL_BASE
    with odom_lock:
        th_mid        = odom["theta"] + dth / 2.0
        odom["x"]    += dist * math.sin(th_mid) * 100   # en cm
        odom["y"]    += dist * math.cos(th_mid) * 100
        odom["theta"] = _wrap(odom["theta"] + dth)

# ═══════════════════════════════════════════════════
# METHODE 2 : ULTRASONS (cap + distance)
# Correction simple : si le robot va droit et que les
# ultrasons latéraux sont symétriques, il est centré.
# On accumule la distance parcourue avec correction.
# ═══════════════════════════════════════════════════
us_pos = {"x": 0.0, "y": 0.0, "theta": 0.0, "active": False, "dist_total": 0.0}
us_lock = threading.Lock()
_us_tl_prev = 0
_us_tr_prev = 0

def us_reset(x=0.0, y=0.0, theta=0.0):
    global _us_tl_prev, _us_tr_prev
    with us_lock:
        us_pos["x"] = x; us_pos["y"] = y; us_pos["theta"] = theta
        us_pos["active"] = True; us_pos["dist_total"] = 0.0
    with pico_lock:
        _us_tl_prev = pico_state["ticks_l"]
        _us_tr_prev = pico_state["ticks_r"]

def us_update():
    global _us_tl_prev, _us_tr_prev
    with pico_lock:
        tl = pico_state["ticks_l"]
        tr = pico_state["ticks_r"]
        dl_us = pico_state["dist_l"]
        dr_us = pico_state["dist_r"]
    dtl = tl - _us_tl_prev
    dtr = tr - _us_tr_prev
    _us_tl_prev = tl
    _us_tr_prev = tr
    if dtl == 0 and dtr == 0:
        return
    dl   = dtl * M_PER_TICK
    dr   = dtr * M_PER_TICK
    dist = (dl + dr) / 2.0
    dth  = (dr - dl) / WHEEL_BASE
    # Correction latérale ultrason : si dl_us et dr_us valides
    # et symétriques, on corrige légèrement le cap
    if dl_us < 200 and dr_us < 200:
        biais_us = (dl_us - dr_us) / (dl_us + dr_us + 1e-6)
        dth     += biais_us * 0.02   # correction douce
    with us_lock:
        th_mid           = us_pos["theta"] + dth / 2.0
        us_pos["x"]     += dist * math.sin(th_mid) * 100
        us_pos["y"]     += dist * math.cos(th_mid) * 100
        us_pos["theta"]  = _wrap(us_pos["theta"] + dth)
        us_pos["dist_total"] += abs(dist) * 100

# ═══════════════════════════════════════════════════
# METHODE 3 : VISION ARUCO
# Position calculée depuis les markers ArUco connus
# ═══════════════════════════════════════════════════
vision_pos = {"x": None, "y": None, "theta": 0.0, "active": False,
              "last_update": 0, "confidence": 0.0}
vision_lock = threading.Lock()
aruco_room  = {}   # {id: {x,y,z}} en cm — markers connus

CAM_W = 640; CAM_H = 480; FOCAL = 800.0
cam_mat = np.array([[FOCAL,0,CAM_W/2],[0,FOCAL,CAM_H/2],[0,0,1]], dtype=np.float32)
dist_co = np.zeros((5,1))

def vision_reset():
    with vision_lock:
        vision_pos["active"] = True
        vision_pos["last_update"] = 0

def vision_update_from_aruco(markers_visible):
    """Calcule la position depuis les ArUco visibles."""
    usable = []
    for mid, data in markers_visible.items():
        key = str(mid)
        if key in aruco_room and "angle" in data:
            pos = aruco_room[key]
            usable.append({"mx":pos["x"],"my":pos["y"],
                           "dist":data["distance"]*100,"angle":data["angle"]})
    if not usable:
        return
    with vision_lock:
        h = vision_pos["theta"]
    xs, ys, ws = [], [], []
    for m in usable:
        d2m = h + m["angle"]
        xs.append(m["mx"] - m["dist"] * math.sin(d2m))
        ys.append(m["my"] - m["dist"] * math.cos(d2m))
        ws.append(1.0 / max(m["dist"], 1.0))
    tw = sum(ws)
    rx = sum(x*w for x,w in zip(xs,ws)) / tw
    ry = sum(y*w for y,w in zip(ys,ws)) / tw
    with vision_lock:
        vision_pos["x"]           = rx
        vision_pos["y"]           = ry
        vision_pos["last_update"] = time.time()
        vision_pos["confidence"]  = min(len(usable)/3.0, 1.0)

# ═══════════════════════════════════════════════════
# METHODE 4 : FUSION EKF (odom + IMU + ArUco)
# ═══════════════════════════════════════════════════
ekf = {"x": 0.0, "y": 0.0, "theta": 0.0,
       "P": np.eye(3)*0.01, "active": False}
ekf_lock = threading.Lock()
_ekf_tl_prev = 0
_ekf_tr_prev = 0
_ekf_imu_ref = None

Q_XY=2e-4; Q_TH=8e-4; R_IMU=0.003; R_ARUCO_M=0.04

def ekf_reset(x=0.0, y=0.0, theta=0.0):
    global _ekf_tl_prev, _ekf_tr_prev, _ekf_imu_ref
    with ekf_lock:
        ekf["x"]=x; ekf["y"]=y; ekf["theta"]=theta
        ekf["P"]=np.eye(3)*0.005; ekf["active"]=True
    with pico_lock:
        _ekf_tl_prev = pico_state["ticks_l"]
        _ekf_tr_prev = pico_state["ticks_r"]
        _ekf_imu_ref = pico_state["yaw"]

def ekf_predict_step():
    global _ekf_tl_prev, _ekf_tr_prev
    with pico_lock:
        tl = pico_state["ticks_l"]
        tr = pico_state["ticks_r"]
    dtl = tl - _ekf_tl_prev
    dtr = tr - _ekf_tr_prev
    _ekf_tl_prev = tl
    _ekf_tr_prev = tr
    if dtl == 0 and dtr == 0:
        return
    dl   = dtl * M_PER_TICK * 100   # cm
    dr   = dtr * M_PER_TICK * 100
    dist = (dl + dr) / 2.0
    dth  = (dr - dl) / (WHEEL_BASE * 100)
    with ekf_lock:
        th_mid = ekf["theta"] + dth/2.0
        ekf["x"]     += dist * math.sin(th_mid)
        ekf["y"]     += dist * math.cos(th_mid)
        ekf["theta"]  = _wrap(ekf["theta"] + dth)
        F = np.array([[1,0,dist*math.cos(th_mid)],[0,1,-dist*math.sin(th_mid)],[0,0,1]])
        q = abs(dist)+1e-6
        Q = np.diag([Q_XY*q,Q_XY*q,Q_TH*q])
        ekf["P"] = F@ekf["P"]@F.T + Q

def ekf_update_imu_step():
    global _ekf_imu_ref
    with pico_lock:
        yaw = pico_state["yaw"]
        imu_ok = pico_state["imu_ok"]
    if not imu_ok or _ekf_imu_ref is None:
        return
    delta = _wrap(math.radians(yaw - _ekf_imu_ref))
    with ekf_lock:
        H = np.array([[0.,0.,1.]])
        R = np.array([[R_IMU]])
        S = H@ekf["P"]@H.T + R
        K = ekf["P"]@H.T@np.linalg.inv(S)
        inn = _wrap(delta - ekf["theta"])
        st  = np.array([ekf["x"],ekf["y"],ekf["theta"]]) + K.flatten()*inn
        ekf["x"]=st[0]; ekf["y"]=st[1]; ekf["theta"]=_wrap(st[2])
        ekf["P"] = (np.eye(3)-K@H)@ekf["P"]

def ekf_update_aruco_step(rx, ry, n=1):
    with ekf_lock:
        H = np.array([[1.,0.,0.],[0.,1.,0.]])
        R = np.eye(2)*(R_ARUCO_M/n)*(100**2)  # cm²
        S = H@ekf["P"]@H.T + R
        K = ekf["P"]@H.T@np.linalg.inv(S)
        inn = np.array([rx-ekf["x"], ry-ekf["y"]])
        st  = np.array([ekf["x"],ekf["y"],ekf["theta"]]) + K@inn
        ekf["x"]=st[0]; ekf["y"]=st[1]; ekf["theta"]=_wrap(st[2])
        ekf["P"] = (np.eye(3)-K@H)@ekf["P"]

# ═══════════════════════════════════════════════════
# THREAD LOCALISATION GLOBAL (50Hz)
# ═══════════════════════════════════════════════════
def loc_thread():
    while True:
        time.sleep(0.02)
        with odom_lock:
            if odom["active"]: odom_update()
        with us_lock:
            if us_pos["active"]: us_update()
        with ekf_lock:
            if ekf["active"]:
                ekf_predict_step()
                ekf_update_imu_step()
        # Marque distance parcourue pour auto-stop
        with us_lock:
            dist_tot = us_pos["dist_total"]
        # session update
        pass

threading.Thread(target=loc_thread, daemon=True).start()

# ═══════════════════════════════════════════════════
# PICO SERIAL
# ═══════════════════════════════════════════════════
PICO_PORT="dev/ttyACM0"; PICO_BAUD=115200
pico_serial=None

def connect_pico():
    global pico_serial
    try:
        pico_serial=serial.Serial("/"+PICO_PORT,PICO_BAUD,timeout=0.1)
        time.sleep(1.5)
        with pico_lock: pico_state["connected"]=True
        print(f"Pico OK")
    except Exception as e:
        print(f"Pico: {e}")
        with pico_lock: pico_state["connected"]=False

def pico_reader():
    print("picored")
    global pico_serial
    while True:
        try:
            if pico_serial and pico_serial.is_open:
                line=pico_serial.readline().decode("utf-8",errors="ignore").strip()
                if line.startswith("DATA:"):
                    print("line")
                    print(line)
                    parts=line.split(":")
                    d={}
                    for i in range(1,len(parts)-1,2):
                        print('tamere')
                        try: d[parts[i]]=parts[i+1]
                        except: pass
                    with pico_lock:
                        pico_state["dist_c"]=float(d.get("C",999))
                        pico_state["dist_l"]=float(d.get("L",999))
                        pico_state["dist_r"]=float(d.get("R",999))
                        pico_state["ticks_l"]=int(d.get("TL",0))
                        pico_state["ticks_r"]=int(d.get("TR",0))
                        pico_state["yaw"]=float(d.get("YAW",0.0))
                        pico_state["imu_ok"]=int(d.get("IMU",0))==1
                        pico_state["last_update"]=time.time()
                        pico_state["connected"]=True
                elif line=="PICO_READY":
                    print('conn')
                    with pico_lock: pico_state["connected"]=True
            else:
                print('hello')
                time.sleep(1); connect_pico()
        except Exception as e:
            print(f"Pico read: {e}")
            with pico_lock: pico_state["connected"]=False
            time.sleep(2); connect_pico()

def send_cmd(cmd):
    try:
        if pico_serial and pico_serial.is_open:
            pico_serial.write((cmd.strip()+"\n").encode())
            with pico_lock: pico_state["nav_state"]=cmd.strip()
    except: pass

connect_pico()
threading.Thread(target=pico_reader,daemon=True).start()

# ═══════════════════════════════════════════════════
# CAMERA + ARUCO
# ═══════════════════════════════════════════════════
latest_frame=None; frame_lock=threading.Lock()
detected_markers={}; detected_lock=threading.Lock()

if CAMERA_AVAILABLE:
    picam2=Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"size":(CAM_W,CAM_H),"format":"RGB888"}))
    picam2.start(); time.sleep(2)
    adict=aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aparams=aruco.DetectorParameters()

def capture_loop():
    global latest_frame
    while True:
        if not CAMERA_AVAILABLE: time.sleep(1); continue
        frame=picam2.capture_array()
        fc=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        now=time.time(); new_det={}
        detector = aruco.ArucoDetector(adict, aparams)
        corners,ids,_ = detector.detectMarkers(fc)
        if ids is not None:
            aruco.drawDetectedMarkers(fc,corners,ids)
            rvecs,tvecs,_=aruco.estimatePoseSingleMarkers(corners,0.17,cam_mat,dist_co)
            for i in range(len(ids)):
                mid=int(ids[i][0]); dist=float(np.linalg.norm(tvecs[i][0]))
                ang=math.atan2(float(tvecs[i][0][0]),float(tvecs[i][0][2]))
                new_det[mid]={"distance":dist,"angle":ang,"timestamp":now}
                c=corners[i][0][0]
                known=str(mid) in aruco_room
                col=(30,200,80) if known else (30,120,255)
                cv2.putText(fc,f"#{mid} {dist:.2f}m",(int(c[0]),int(c[1]-10)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1)
        with detected_lock:
            detected_markers.clear(); detected_markers.update(new_det)
        # Update vision + EKF depuis ArUco
        if new_det:
            vision_update_from_aruco(new_det)
            # Calcule position ArUco pour EKF
            with vision_lock:
                vx,vy = vision_pos["x"],vision_pos["y"]
            if vx is not None:
                with ekf_lock:
                    if ekf["active"]:
                        ekf_update_aruco_step(vx,vy,n=len(new_det))

        # Overlay info
        h,w=fc.shape[:2]
        with pico_lock: ns=pico_state["nav_state"]; pok=pico_state["connected"]
        with odom_lock: ox,oy=odom["x"],odom["y"]
        with ekf_lock: ex,ey=ekf["x"],ekf["y"]
        cv2.rectangle(fc,(0,0),(w,22),(18,22,30),-1)
        info=f"ODOM({ox:.1f},{oy:.1f})cm  EKF({ex:.1f},{ey:.1f})cm  {ns}"
        cv2.putText(fc,info,(5,14),cv2.FONT_HERSHEY_SIMPLEX,0.34,(180,200,220),1)

        img=Image.fromarray(cv2.cvtColor(fc,cv2.COLOR_BGR2RGB))
        buf=io.BytesIO(); img.save(buf,format="JPEG",quality=78)
        with frame_lock: latest_frame=buf.getvalue()
        time.sleep(0.04)

threading.Thread(target=capture_loop,daemon=True).start()

def stream_cam():
    while True:
        with frame_lock: f=latest_frame
        if f: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+f+b"\r\n"
        time.sleep(0.04)

@app.get("/video")
def video(): return StreamingResponse(stream_cam(),media_type="multipart/x-mixed-replace; boundary=frame")

# ═══════════════════════════════════════════════════
# API ETAT
# ═══════════════════════════════════════════════════
@app.get("/state")
def get_state():
    with pico_lock: ps=dict(pico_state)
    with odom_lock: od=dict(odom)
    with us_lock:   us=dict(us_pos)
    with vision_lock: vi=dict(vision_pos)
    with ekf_lock:  ek={"x":ekf["x"],"y":ekf["y"],"theta":math.degrees(ekf["theta"]),"active":ekf["active"]}
    with detected_lock: det=dict(detected_markers)
    return {
        "pico": ps, "odom": od, "us": us, "vision": vi, "ekf": ek,
        "markers_visible": len(det),
        "imu_ok": ps["imu_ok"], "imu_yaw": ps["yaw"],
        "session": {k:v for k,v in session.items() if k!="log"},
        "session_log": session["log"][-10:],
    }

# ═══════════════════════════════════════════════════
# API CONTROLE ROBOT
# ═══════════════════════════════════════════════════
@app.post("/cmd")
async def api_cmd(req:Request):
    d=await req.json(); send_cmd(d.get("cmd","STOP"))
    return {"status":"ok"}

@app.post("/reset_all")
async def api_reset(req:Request):
    d=await req.json()
    x=float(d.get("x",0)); y=float(d.get("y",0))
    odom_reset(x,y); us_reset(x,y); vision_reset(); ekf_reset(x,y)
    session["last_stop_dist"]=0
    return {"status":"ok"}

@app.post("/set_aruco_marker")
async def api_set_marker(req:Request):
    d=await req.json()
    mid=str(d.get("id"))
    aruco_room[mid]={"x":float(d.get("x",0)),"y":float(d.get("y",0)),"z":float(d.get("z",0))}
    return {"status":"ok","markers":aruco_room}

@app.get("/get_aruco_markers")
def api_get_markers(): return aruco_room

# ═══════════════════════════════════════════════════
# API EXPERIENCE
# ═══════════════════════════════════════════════════
@app.post("/exp/start")
async def exp_start(req:Request):
    d=await req.json()
    session["running"]  = True
    session["mode"]     = d.get("mode","position")
    session["start_time"]= time.time()
    session["step"]     = 0
    session["target_x"] = float(d.get("target_x",100))
    session["target_y"] = float(d.get("target_y",0))
    session["auto_stop_distance"] = float(d.get("stop_every",50))
    session["log"]      = []
    session["last_stop_dist"] = 0
    odom_reset(); us_reset(); vision_reset(); ekf_reset()
    return {"status":"ok"}

@app.post("/exp/stop")
async def exp_stop():
    session["running"]=False
    send_cmd("STOP")
    return {"status":"ok"}

@app.get("/exp/snapshot")
def exp_snapshot():
    """
    Capture instantanée des 4 méthodes.
    Retourne les positions estimées par chaque méthode.
    """
    t = time.time() - session["start_time"] if session["running"] else 0
    with odom_lock: ox,oy=odom["x"],odom["y"]
    with us_lock:   ux,uy=us_pos["x"],us_pos["y"]
    with vision_lock: vx=(vision_pos["x"] or 0); vy=(vision_pos["y"] or 0)
    with ekf_lock:  fx,fy=ekf["x"],ekf["y"]
    return {
        "timestamp": round(t,2),
        "odom":   {"x":round(ox,2),"y":round(oy,2)},
        "us":     {"x":round(ux,2),"y":round(uy,2)},
        "vision": {"x":round(vx,2),"y":round(vy,2)},
        "fusion": {"x":round(fx,2),"y":round(fy,2)},
    }

@app.post("/exp/record")
async def exp_record(req:Request):
    """
    Enregistre une mesure avec la vraie position fournie par l'utilisateur.
    Mode position ou trajectoire : ajoute dans methods et trajectories.
    Mode temps : ajoute dans time_series.
    """
    d=await req.json()
    real_x = float(d.get("real_x",0))
    real_y = float(d.get("real_y",0))
    mode   = d.get("mode","position")
    snap   = d.get("snapshot",{})
    t      = float(d.get("t",0))

    with exp_lock:
        data = load_exp_data()

        methods_map = {
            "odom":   (snap.get("odom",{}).get("x",0),   snap.get("odom",{}).get("y",0)),
            "ultrason":(snap.get("us",{}).get("x",0),    snap.get("us",{}).get("y",0)),
            "vision": (snap.get("vision",{}).get("x",0), snap.get("vision",{}).get("y",0)),
            "fusion": (snap.get("fusion",{}).get("x",0), snap.get("fusion",{}).get("y",0)),
        }

        for mname,(ex,ey) in methods_map.items():
            error = math.sqrt((real_x-ex)**2+(real_y-ey)**2)
            if mode=="position":
                data["methods"].setdefault(mname,[]).append(error)
            elif mode=="time":
                data["time_series"].setdefault(mname,[]).append((round(t,2),round(error,2)))
            elif mode=="trajectory":
                data["trajectories"].setdefault(mname,[]).append(
                    (round(real_x,2),round(real_y,2),round(ex,2),round(ey,2)))

        save_exp_data(data)
        session["step"] += 1
        session["log"].append({
            "step": session["step"],
            "real": {"x":real_x,"y":real_y},
            "snap": snap,
            "mode": mode,
            "t":    t,
        })

    return {"status":"ok","step":session["step"]}

@app.get("/exp/data")
def exp_get_data():
    with exp_lock: return load_exp_data()

@app.post("/exp/clear")
async def exp_clear():
    with exp_lock:
        d={"methods":{},"time_series":{},"trajectories":{}}
        save_exp_data(d)
    return {"status":"ok"}

@app.get("/exp/export")
def exp_export():
    """Télécharger robot_data.json directement."""
    from fastapi.responses import FileResponse
    if os.path.exists(DATA_FILE):
        return FileResponse(DATA_FILE,media_type="application/json",
                           filename="robot_data.json")
    return JSONResponse({"error":"no data"},status_code=404)

# ═══════════════════════════════════════════════════
# INTERFACE WEB
# ═══════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot v9 — Collecte</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,600;0,900;1,300;1,600&display=swap" rel="stylesheet">
<style>
:root {
  --ink:   #12140f;
  --paper: #f5f2eb;
  --cream: #ede9df;
  --line:  #d4cfc4;
  --green: #2d6a2d;
  --red:   #b83232;
  --blue:  #1a4a8a;
  --amber: #8a5a1a;
  --mono:  'DM Mono', monospace;
  --serif: 'Fraunces', serif;
}
* { box-sizing:border-box; margin:0; padding:0; }
body {
  background: var(--paper);
  color: var(--ink);
  font-family: var(--mono);
  font-size: 12px;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* HEADER */
header {
  display: flex; align-items: center; gap: 16px;
  padding: 0 20px; height: 52px;
  border-bottom: 2px solid var(--ink);
  background: var(--ink); color: var(--paper);
  flex-shrink: 0;
}
.logo-text {
  font-family: var(--serif); font-size: 18px;
  font-weight: 900; letter-spacing: -.01em;
  color: var(--paper);
}
.logo-text span { font-style: italic; color: #c8e6c9; }
.hchip {
  padding: 2px 8px; border-radius: 3px;
  font-size: 10px; font-weight: 500;
  border: 1px solid rgba(255,255,255,.2);
  color: rgba(255,255,255,.7);
}
.hchip.ok  { border-color: #4caf50; color: #4caf50; }
.hchip.err { border-color: #ef5350; color: #ef5350; }
.hchip.imu-ok  { border-color: #64b5f6; color: #64b5f6; }

/* MAIN GRID */
.main {
  display: grid;
  grid-template-columns: 320px 1fr 300px;
  flex: 1; overflow: hidden; min-height: 0;
}

/* COL — shared */
.col {
  display: flex; flex-direction: column;
  border-right: 1px solid var(--line);
  overflow-y: auto; min-height: 0;
}
.col:last-child { border-right: none; }

.sec-title {
  font-family: var(--serif); font-size: 11px;
  font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: #888;
  padding: 12px 14px 4px;
  border-bottom: 1px solid var(--line);
  position: sticky; top: 0; background: var(--paper); z-index: 1;
}

/* VIDEO */
.vid-wrap {
  position: relative; background: #1a1a1a; flex-shrink: 0;
  border-bottom: 1px solid var(--line);
}
.vid-wrap img { width:100%; display:block; max-height:200px; object-fit:cover; }
.vid-overlay {
  position: absolute; bottom: 6px; left: 6px; right: 6px;
  display: flex; justify-content: space-between;
  pointer-events: none;
}
.vid-badge {
  background: rgba(0,0,0,.65); color: #fff;
  font-size: 9px; padding: 2px 6px; border-radius: 2px;
}

/* STATUS GRID */
.status-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 1px; background: var(--line); border-bottom: 1px solid var(--line);
  flex-shrink: 0;
}
.sc {
  background: var(--paper); padding: 8px 10px;
  display: flex; flex-direction: column; gap: 2px;
}
.sc-label { font-size: 9px; color: #999; letter-spacing: .06em; text-transform: uppercase; }
.sc-value { font-size: 16px; font-weight: 500; transition: color .2s; }
.sc-sub   { font-size: 9px; color: #999; }
.v-ok  { color: var(--green); }
.v-mid { color: var(--amber); }
.v-bad { color: var(--red);   }

/* METHOD TABLE */
.method-table {
  flex-shrink: 0;
  border-bottom: 1px solid var(--line);
}
.mt-row {
  display: grid; grid-template-columns: 90px 1fr 1fr;
  gap: 0; border-bottom: 1px solid var(--line);
  align-items: center;
}
.mt-row:last-child { border-bottom: none; }
.mt-name {
  padding: 7px 10px;
  font-weight: 500; font-size: 11px;
  border-right: 1px solid var(--line);
}
.mt-val {
  padding: 6px 8px; font-size: 10px;
  font-family: var(--mono); text-align: center;
  border-right: 1px solid var(--line);
}
.mt-val:last-child { border-right: none; }
.mt-header { background: var(--cream); }
.mt-header .mt-name,.mt-header .mt-val {
  font-size: 9px; color: #999; text-transform: uppercase;
  letter-spacing:.06em; font-weight:400;
}

/* CONTROLS */
.ctrl-grid {
  display: grid; grid-template-columns: repeat(3,1fr);
  gap: 4px; padding: 10px; flex-shrink: 0;
  border-bottom: 1px solid var(--line);
}
.btn {
  padding: 8px 4px; border: 1.5px solid var(--line);
  background: var(--paper); color: var(--ink);
  font-family: var(--mono); font-size: 10px; font-weight: 500;
  cursor: pointer; border-radius: 4px; text-align: center;
  transition: all .12s; line-height: 1.4;
}
.btn:hover { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.btn:active { transform: scale(.95); }
.btn.stop { border-color: var(--red); color: var(--red); background: #fef5f5; }
.btn.stop:hover { background: var(--red); color: #fff; }
.btn.go   { border-color: var(--green); color: var(--green); background: #f0faf0; }
.btn.go:hover { background: var(--green); color: #fff; }
.btn.primary { background: var(--ink); color: var(--paper); border-color: var(--ink); }
.btn.primary:hover { background: #333; }
.btn.danger { border-color: var(--red); color: var(--red); }
.btn.danger:hover { background: var(--red); color: #fff; }
.btn-empty { background: transparent; border: none; pointer-events: none; }

/* EXPERIMENT PANEL (centre) */
.exp-panel {
  padding: 14px;
}
.exp-title {
  font-family: var(--serif); font-size: 22px;
  font-weight: 900; margin-bottom: 4px;
}
.exp-sub {
  font-size: 10px; color: #888; margin-bottom: 14px;
}

.mode-tabs {
  display: flex; gap: 0; border: 1.5px solid var(--ink);
  border-radius: 5px; overflow: hidden; margin-bottom: 14px;
}
.mode-tab {
  flex: 1; padding: 6px 8px; text-align: center;
  font-size: 10px; font-weight: 500; cursor: pointer;
  border: none; background: var(--paper); color: #888;
  transition: all .15s; font-family: var(--mono);
}
.mode-tab.active { background: var(--ink); color: var(--paper); }

.field-row {
  display: grid; grid-template-columns: repeat(auto-fill,minmax(100px,1fr));
  gap: 8px; margin-bottom: 10px;
}
.field {
  display: flex; flex-direction: column; gap: 3px;
}
.field label {
  font-size: 9px; color: #888; text-transform: uppercase;
  letter-spacing: .06em;
}
.field input {
  background: var(--cream); border: 1.5px solid var(--line);
  color: var(--ink); border-radius: 4px;
  padding: 6px 8px; font-family: var(--mono); font-size: 12px;
  outline: none; width: 100%;
}
.field input:focus { border-color: var(--ink); }

.exp-actions {
  display: flex; gap: 6px; margin-bottom: 14px; flex-wrap: wrap;
}

/* SNAPSHOT BOX */
.snap-box {
  border: 1.5px solid var(--line); border-radius: 5px;
  overflow: hidden; margin-bottom: 12px;
}
.snap-header {
  background: var(--cream); border-bottom: 1px solid var(--line);
  padding: 6px 10px; font-size: 10px; color: #888;
  display: flex; justify-content: space-between; align-items: center;
}
.snap-grid {
  display: grid; grid-template-columns: repeat(4,1fr);
  gap: 1px; background: var(--line);
}
.snap-cell {
  background: var(--paper); padding: 8px 8px;
  display: flex; flex-direction: column; gap: 2px; align-items: center;
}
.snap-method { font-size: 9px; color: #888; text-transform: uppercase; }
.snap-val    { font-size: 13px; font-weight: 500; font-family: var(--mono); }

/* REAL POS INPUT */
.real-pos-box {
  border: 1.5px solid var(--amber); border-radius: 5px;
  padding: 10px; margin-bottom: 12px; background: #fffbf5;
}
.real-pos-title {
  font-size: 10px; font-weight: 500; color: var(--amber);
  text-transform: uppercase; letter-spacing: .06em; margin-bottom: 8px;
}
.real-pos-row {
  display: flex; gap: 8px; align-items: flex-end;
}

/* LOG */
.log-box {
  border: 1.5px solid var(--line); border-radius: 5px;
  overflow: hidden;
}
.log-header {
  background: var(--cream); padding: 5px 10px;
  font-size: 9px; color: #888; border-bottom: 1px solid var(--line);
  display: flex; justify-content: space-between;
}
.log-entry {
  padding: 5px 10px; border-bottom: 1px solid var(--line);
  font-size: 10px; display: flex; gap: 10px; align-items: center;
}
.log-entry:last-child { border-bottom: none; }
.log-step { color: #888; width: 28px; flex-shrink: 0; }
.log-method-vals { flex: 1; display: grid; grid-template-columns: repeat(4,1fr); gap: 4px; }
.log-err { font-size: 9px; text-align: center; }
.log-err span { color: var(--green); font-weight: 500; }
.log-err.warn span { color: var(--amber); }
.log-err.bad  span { color: var(--red); }

/* RIGHT COL — data recap */
.data-recap {
  padding: 10px;
}
.method-card {
  border: 1.5px solid var(--line); border-radius: 5px;
  overflow: hidden; margin-bottom: 8px;
}
.mc-header {
  padding: 6px 10px; font-size: 10px; font-weight: 500;
  border-bottom: 1px solid var(--line); background: var(--cream);
  display: flex; justify-content: space-between;
}
.mc-stats {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 0; background: var(--line);
}
.mc-stat {
  background: var(--paper); padding: 6px 10px;
}
.mc-stat-l { font-size: 9px; color: #888; }
.mc-stat-v { font-size: 14px; font-weight: 500; font-family: var(--mono); }
.mc-count  { font-size: 9px; color: #888; }

.export-section { padding: 10px; border-top: 1px solid var(--line); }
.export-btn {
  width: 100%; padding: 10px; border: 1.5px solid var(--green);
  background: #f0faf0; color: var(--green); border-radius: 5px;
  font-family: var(--mono); font-size: 11px; font-weight: 500;
  cursor: pointer; transition: all .15s; margin-bottom: 6px;
}
.export-btn:hover { background: var(--green); color: #fff; }
.clear-btn {
  width: 100%; padding: 7px; border: 1.5px solid var(--line);
  background: var(--paper); color: #888; border-radius: 5px;
  font-family: var(--mono); font-size: 10px;
  cursor: pointer; transition: all .15s;
}
.clear-btn:hover { border-color: var(--red); color: var(--red); }

/* ARUCO CONFIG */
.aruco-sec { padding: 10px; border-top: 1px solid var(--line); }
.aruco-row {
  display: flex; gap: 5px; align-items: flex-end; margin-top: 6px;
}
.aruco-marker-list { margin-top: 6px; }
.am-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 4px 0; border-bottom: 1px solid var(--line); font-size: 10px;
}
.am-item:last-child { border-bottom: none; }

/* TOAST */
.toast {
  position: fixed; bottom: 16px; left: 50%;
  transform: translateX(-50%) translateY(10px);
  background: var(--ink); color: var(--paper);
  padding: 7px 16px; border-radius: 4px; font-size: 11px;
  opacity: 0; pointer-events: none; transition: all .2s; z-index: 200;
}
.toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }

.status-bar {
  display: flex; align-items: center; gap: 10px;
  padding: 5px 14px; border-top: 1px solid var(--line);
  background: var(--cream); font-size: 9px; color: #888;
  flex-shrink: 0; position: sticky; bottom: 0;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #ccc; }
.status-dot.on { background: var(--green); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1}50%{opacity:.4} }
</style>
</head>
<body>

<header>
  <div class="logo-text">Robot <span>Nav</span> v9</div>
  <div class="hchip" id="h-pico">Pico...</div>
  <div class="hchip" id="h-imu">IMU...</div>
  <div class="hchip" id="h-cam">Cam...</div>
  <div style="margin-left:auto;font-size:10px;color:rgba(255,255,255,.5)" id="clock"></div>
</header>

<div class="main">

<!-- ══ COL GAUCHE — robot state ══ -->
<div class="col">
  <div class="vid-wrap">
    <img src="/video" alt="Camera">
    <div class="vid-overlay">
      <div class="vid-badge" id="vid-nav">STOP</div>
      <div class="vid-badge" id="vid-markers">0 ArUco</div>
    </div>
  </div>

  <div class="sec-title">Capteurs</div>
  <div class="status-grid">
    <div class="sc"><div class="sc-label">US Gauche</div><div class="sc-value v-ok" id="dl">--</div><div class="sc-sub">cm</div></div>
    <div class="sc"><div class="sc-label">US Centre</div><div class="sc-value v-ok" id="dc">--</div><div class="sc-sub">cm</div></div>
    <div class="sc"><div class="sc-label">US Droite</div><div class="sc-value v-ok" id="dr">--</div><div class="sc-sub">cm</div></div>
    <div class="sc"><div class="sc-label">IMU Yaw</div><div class="sc-value" id="imu-yaw" style="color:var(--blue)">--</div><div class="sc-sub">deg</div></div>
  </div>

  <div class="sec-title">Position estimee (cm)</div>
  <div class="method-table">
    <div class="mt-row mt-header">
      <div class="mt-name">Methode</div>
      <div class="mt-val">X</div>
      <div class="mt-val">Y</div>
    </div>
    <div class="mt-row">
      <div class="mt-name">Odometrie</div>
      <div class="mt-val" id="ox">--</div>
      <div class="mt-val" id="oy">--</div>
    </div>
    <div class="mt-row">
      <div class="mt-name">Ultrasons</div>
      <div class="mt-val" id="ux">--</div>
      <div class="mt-val" id="uy">--</div>
    </div>
    <div class="mt-row">
      <div class="mt-name">Vision</div>
      <div class="mt-val" id="vx">--</div>
      <div class="mt-val" id="vy">--</div>
    </div>
    <div class="mt-row" style="background:#f5faf5">
      <div class="mt-name" style="font-weight:600;color:var(--green)">Fusion EKF</div>
      <div class="mt-val" id="fx" style="color:var(--green);font-weight:600">--</div>
      <div class="mt-val" id="fy" style="color:var(--green);font-weight:600">--</div>
    </div>
  </div>

  <div class="sec-title">Controles manuels</div>
  <div class="ctrl-grid">
    <div class="btn-empty"></div>
    <button class="btn go" onclick="cmd('AVANCE')">&#9650;<br>Avant</button>
    <div class="btn-empty"></div>
    <button class="btn" onclick="cmd('PIVOT_G')">&#9664; G</button>
    <button class="btn stop" onclick="cmd('STOP')">Stop</button>
    <button class="btn" onclick="cmd('PIVOT_D')">D &#9654;</button>
    <div class="btn-empty"></div>
    <button class="btn" onclick="cmd('RECUL')">Recul<br>&#9660;</button>
    <div class="btn-empty"></div>
  </div>

  <div class="sec-title">Reinitialiser</div>
  <div style="padding:8px 10px;border-bottom:1px solid var(--line)">
    <div class="field-row" style="margin-bottom:6px">
      <div class="field"><label>X depart (cm)</label><input type="number" id="start-x" value="0"></div>
      <div class="field"><label>Y depart (cm)</label><input type="number" id="start-y" value="0"></div>
    </div>
    <button class="btn primary" style="width:100%;padding:8px" onclick="resetAll()">Reinitialiser toutes les methodes</button>
  </div>

  <div class="status-bar">
    <div class="status-dot" id="status-dot"></div>
    <span id="status-txt">En attente...</span>
    <span style="margin-left:auto" id="ticks-info"></span>
  </div>
</div>

<!-- ══ COL CENTRE — experience ══ -->
<div class="col" style="border-right:1px solid var(--line)">
  <div class="exp-panel">
    <div class="exp-title">Experience</div>
    <div class="exp-sub">Collecte de donnees pour analyse comparative des 4 methodes</div>

    <!-- Mode tabs -->
    <div class="mode-tabs">
      <button class="mode-tab active" onclick="setMode('position',this)">1. Erreur position</button>
      <button class="mode-tab" onclick="setMode('time',this)">2. Erreur / temps</button>
      <button class="mode-tab" onclick="setMode('trajectory',this)">3. Trajectoire</button>
    </div>

    <div id="mode-desc" style="font-size:10px;color:#888;margin-bottom:10px;padding:7px;background:var(--cream);border-radius:4px;border-left:3px solid var(--ink)">
      Mode position : place le robot a un point cible, releve 5 fois la position reelle vs estimee pour chaque methode.
    </div>

    <!-- Config experience -->
    <div class="field-row">
      <div class="field"><label>Cible X (cm)</label><input type="number" id="target-x" value="100" step="5"></div>
      <div class="field"><label>Cible Y (cm)</label><input type="number" id="target-y" value="0" step="5"></div>
      <div class="field"><label>Arreter tous les (cm)</label><input type="number" id="stop-every" value="50" step="10"></div>
    </div>

    <div class="exp-actions">
      <button class="btn go" onclick="startExp()">Demarrer experience</button>
      <button class="btn stop" onclick="stopExp()">Arreter</button>
      <button class="btn" onclick="takeSingleSnapshot()">Snapshot maintenant</button>
    </div>

    <!-- Snapshot -->
    <div class="snap-box">
      <div class="snap-header">
        <span>Positions estimees (cm)</span>
        <span id="snap-time" style="color:#888">--</span>
      </div>
      <div class="snap-grid">
        <div class="snap-cell"><div class="snap-method">Odometrie</div><div class="snap-val" id="s-ox">--</div><div style="font-size:9px;color:#888" id="s-oy">--</div></div>
        <div class="snap-cell"><div class="snap-method">Ultrasons</div><div class="snap-val" id="s-ux">--</div><div style="font-size:9px;color:#888" id="s-uy">--</div></div>
        <div class="snap-cell"><div class="snap-method">Vision</div><div class="snap-val" id="s-vx">--</div><div style="font-size:9px;color:#888" id="s-vy">--</div></div>
        <div class="snap-cell" style="background:#f5faf5"><div class="snap-method" style="color:var(--green)">Fusion</div><div class="snap-val" style="color:var(--green)" id="s-fx">--</div><div style="font-size:9px;color:#888" id="s-fy">--</div></div>
      </div>
    </div>

    <!-- Position reelle -->
    <div class="real-pos-box">
      <div class="real-pos-title">Entrer la position REELLE (mesuree au sol)</div>
      <div class="real-pos-row">
        <div class="field" style="flex:1"><label>X reel (cm)</label><input type="number" id="real-x" placeholder="ex: 100"></div>
        <div class="field" style="flex:1"><label>Y reel (cm)</label><input type="number" id="real-y" placeholder="ex: 0"></div>
        <button class="btn primary" onclick="recordMeasure()" style="padding:6px 14px;align-self:flex-end">Enregistrer</button>
      </div>
    </div>

    <!-- Log -->
    <div class="log-box">
      <div class="log-header">
        <span>Journal des mesures</span>
        <span id="log-count">0 mesures</span>
      </div>
      <div id="log-list" style="max-height:180px;overflow-y:auto">
        <div style="padding:12px;text-align:center;color:#888;font-size:10px">Aucune mesure enregistree</div>
      </div>
    </div>
  </div>

  <!-- Config ArUco -->
  <div class="aruco-sec">
    <div class="sec-title" style="padding:0 0 6px;border-bottom:none;position:static">Balises ArUco (pour la methode Vision)</div>
    <div class="aruco-row">
      <div class="field"><label>ID</label><input type="number" id="a-id" placeholder="0" style="width:60px"></div>
      <div class="field"><label>X (cm)</label><input type="number" id="a-x" placeholder="0"></div>
      <div class="field"><label>Y (cm)</label><input type="number" id="a-y" placeholder="0"></div>
      <button class="btn primary" onclick="addAruco()" style="padding:6px 10px;align-self:flex-end">+</button>
    </div>
    <div class="aruco-marker-list" id="aruco-list">
      <div style="color:#888;font-size:10px;padding:4px 0">Aucun marqueur configure</div>
    </div>
  </div>
</div>

<!-- ══ COL DROITE — recap donnees ══ -->
<div class="col">
  <div class="sec-title">Recap donnees collectees</div>
  <div class="data-recap" id="data-recap">
    <div style="color:#888;font-size:10px;text-align:center;padding:20px">
      Aucune donnee.<br>Commencez une experience.
    </div>
  </div>

  <div class="export-section">
    <button class="export-btn" onclick="exportData()">
      Telecharger robot_data.json
    </button>
    <div style="font-size:9px;color:#888;margin-bottom:8px;text-align:center">
      Compatible avec votre script analyse Python
    </div>
    <button class="clear-btn" onclick="clearData()">Effacer toutes les donnees</button>
  </div>
</div>

</div><!-- .main -->

<div class="toast" id="toast"></div>

<script>
let currentMode = 'position';
let currentSnap = null;
let sessionRunning = false;
let expStartTime  = null;

const modeDescs = {
  position: "Mode position : place le robot a un point cible, releve 5 fois la position reelle vs estimee pour chaque methode. (menu 1 du script Python)",
  time:     "Mode temps : fais avancer le robot, note l'erreur a 5 instants differents. (menu 2 du script Python)",
  trajectory: "Mode trajectoire : releve plusieurs points pendant un trajet. (menu 3 du script Python)",
};

// Clock
setInterval(() => document.getElementById('clock').textContent = new Date().toLocaleTimeString('fr-FR'), 1000);

// Toast
function toast(m, dur=2200) {
  const t = document.getElementById('toast');
  t.textContent = m; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), dur);
}

// Mode
function setMode(mode, btn) {
  currentMode = mode;
  document.querySelectorAll('.mode-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('mode-desc').textContent = modeDescs[mode];
}

// Commands
async function cmd(c) {
  await fetch('/cmd', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({cmd:c}) });
}

async function resetAll() {
  const x = +document.getElementById('start-x').value || 0;
  const y = +document.getElementById('start-y').value || 0;
  await fetch('/reset_all', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({x,y}) });
  toast(`Toutes les methodes reinitalisees a (${x}, ${y}) cm`);
}

// Experiment
async function startExp() {
  sessionRunning = true;
  expStartTime   = Date.now();
  const tx = +document.getElementById('target-x').value || 100;
  const ty = +document.getElementById('target-y').value || 0;
  const se = +document.getElementById('stop-every').value || 50;
  await fetch('/exp/start', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ mode: currentMode, target_x: tx, target_y: ty, stop_every: se })
  });
  toast('Experience demarree !');
  await takeSingleSnapshot();
}

async function stopExp() {
  sessionRunning = false;
  await fetch('/exp/stop', { method:'POST' });
  toast('Experience arretee');
}

async function takeSingleSnapshot() {
  const res  = await fetch('/exp/snapshot');
  const snap = await res.json();
  currentSnap = snap;
  document.getElementById('snap-time').textContent = `t = ${snap.timestamp}s`;
  document.getElementById('s-ox').textContent = snap.odom.x.toFixed(1);
  document.getElementById('s-oy').textContent = `y: ${snap.odom.y.toFixed(1)}`;
  document.getElementById('s-ux').textContent = snap.us.x.toFixed(1);
  document.getElementById('s-uy').textContent = `y: ${snap.us.y.toFixed(1)}`;
  document.getElementById('s-vx').textContent = snap.vision.x.toFixed(1);
  document.getElementById('s-vy').textContent = `y: ${snap.vision.y.toFixed(1)}`;
  document.getElementById('s-fx').textContent = snap.fusion.x.toFixed(1);
  document.getElementById('s-fy').textContent = `y: ${snap.fusion.y.toFixed(1)}`;
}

async function recordMeasure() {
  const real_x = parseFloat(document.getElementById('real-x').value);
  const real_y = parseFloat(document.getElementById('real-y').value);
  if (isNaN(real_x) || isNaN(real_y)) { toast('Entre la position reelle !'); return; }
  if (!currentSnap) { await takeSingleSnapshot(); }
  const t = expStartTime ? (Date.now() - expStartTime) / 1000 : 0;
  const res = await fetch('/exp/record', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ real_x, real_y, mode: currentMode, snapshot: currentSnap, t })
  });
  const d = await res.json();
  toast(`Mesure ${d.step} enregistree !`);
  document.getElementById('real-x').value = '';
  document.getElementById('real-y').value = '';
  await takeSingleSnapshot();
  await refreshData();
}

// ArUco
async function addAruco() {
  const id = document.getElementById('a-id').value;
  const x  = +document.getElementById('a-x').value || 0;
  const y  = +document.getElementById('a-y').value || 0;
  if (!id) { toast('ID requis'); return; }
  await fetch('/set_aruco_marker', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ id: +id, x, y, z: 0 })
  });
  toast(`Marker #${id} ajoute`);
  refreshAruco();
}

async function refreshAruco() {
  const d = await (await fetch('/get_aruco_markers')).json();
  const el = document.getElementById('aruco-list');
  const keys = Object.keys(d);
  if (!keys.length) {
    el.innerHTML = '<div style="color:#888;font-size:10px;padding:4px 0">Aucun marqueur configure</div>';
    return;
  }
  el.innerHTML = keys.map(k =>
    `<div class="am-item"><span style="font-weight:500">#${k}</span><span style="color:#888">(${d[k].x}, ${d[k].y}) cm</span></div>`
  ).join('');
}

// Data recap
function calcMean(arr) { return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : null; }
function calcStd(arr) {
  if (!arr.length) return null;
  const m = calcMean(arr);
  return Math.sqrt(arr.reduce((a,b)=>a+(b-m)**2,0)/arr.length);
}

async function refreshData() {
  const d = await (await fetch('/exp/data')).json();
  const el = document.getElementById('data-recap');
  const methods = { odom:'Odometrie', ultrason:'Ultrasons', vision:'Vision', fusion:'Fusion EKF' };
  const colors  = { odom:'var(--ink)', ultrason:'var(--amber)', vision:'var(--blue)', fusion:'var(--green)' };
  let html = '';
  let hasData = false;

  for (const [k, label] of Object.entries(methods)) {
    const vals = (d.methods[k] || []).concat(
      (d.time_series[k] || []).map(v => v[1])
    );
    if (vals.length) hasData = true;
    const mean = calcMean(vals);
    const std  = calcStd(vals);
    const traj = (d.trajectories[k] || []);

    html += `<div class="method-card">
      <div class="mc-header" style="color:${colors[k]}">${label}<span class="mc-count">${vals.length} mesure(s)  ${traj.length} pt traj.</span></div>
      <div class="mc-stats">
        <div class="mc-stat"><div class="mc-stat-l">Erreur moyenne</div><div class="mc-stat-v">${mean!=null?mean.toFixed(1)+'cm':'--'}</div></div>
        <div class="mc-stat"><div class="mc-stat-l">Ecart-type</div><div class="mc-stat-v">${std!=null?std.toFixed(1):'--'}</div></div>
      </div>
    </div>`;
  }

  el.innerHTML = hasData ? html : '<div style="color:#888;font-size:10px;text-align:center;padding:20px">Aucune donnee.<br>Commencez une experience.</div>';
}

// Export
async function exportData() {
  window.location.href = '/exp/export';
  toast('Telechargement en cours...');
}

async function clearData() {
  if (!confirm('Effacer TOUTES les donnees ?')) return;
  await fetch('/exp/clear', { method:'POST' });
  toast('Donnees effacees');
  refreshData();
}

// State update
function svClass(v) { return v < 15 ? 'sc-value v-bad' : v < 35 ? 'sc-value v-mid' : 'sc-value v-ok'; }

async function pollState() {
  try {
    const s = await (await fetch('/state')).json();

    // Pico chip
    const pc = document.getElementById('h-pico');
    pc.textContent = s.pico.connected ? 'Pico OK' : 'Pico OFF';
    pc.className   = 'hchip ' + (s.pico.connected ? 'ok' : 'err');

    // IMU chip
    const ic = document.getElementById('h-imu');
    ic.textContent = s.imu_ok ? 'BNO085 OK' : 'IMU OFF';
    ic.className   = 'hchip ' + (s.imu_ok ? 'imu-ok' : 'err');

    // Cam chip
    const cc = document.getElementById('h-cam');
    cc.textContent = s.markers_visible > 0 ? `${s.markers_visible} ArUco` : 'Cam OK';
    cc.className   = 'hchip ' + (s.markers_visible > 0 ? 'ok' : '');

    // Distances
    ['c','l','r'].forEach(k => {
      const el = document.getElementById(`d${k}`), v = s.pico[`dist_${k}`];
      el.textContent = v >= 999 ? '--' : Math.round(v);
      el.className   = svClass(v);
    });
    document.getElementById('imu-yaw').textContent = s.imu_ok ? s.imu_yaw.toFixed(1) : '--';

    // Positions
    document.getElementById('ox').textContent = s.odom.x.toFixed(1);
    document.getElementById('oy').textContent = s.odom.y.toFixed(1);
    document.getElementById('ux').textContent = s.us.x.toFixed(1);
    document.getElementById('uy').textContent = s.us.y.toFixed(1);
    document.getElementById('vx').textContent = s.vision.x != null ? s.vision.x.toFixed(1) : '--';
    document.getElementById('vy').textContent = s.vision.y != null ? s.vision.y.toFixed(1) : '--';
    document.getElementById('fx').textContent = s.ekf.x.toFixed(1);
    document.getElementById('fy').textContent = s.ekf.y.toFixed(1);

    // Nav state
    document.getElementById('vid-nav').textContent = s.pico.nav_state || 'STOP';
    document.getElementById('vid-markers').textContent = `${s.markers_visible} ArUco`;

    // Status bar
    const dot = document.getElementById('status-dot');
    const txt = document.getElementById('status-txt');
    if (s.session.running) {
      dot.classList.add('on');
      txt.textContent = `Experience en cours — mode ${s.session.mode} — etape ${s.session.step}`;
    } else {
      dot.classList.remove('on');
      txt.textContent = 'En attente';
    }
    document.getElementById('ticks-info').textContent = `TL:${s.pico.ticks_l}  TR:${s.pico.ticks_r}`;

    // Log
    const logEl = document.getElementById('log-list');
    const log   = s.session_log || [];
    document.getElementById('log-count').textContent = `${s.session.step || 0} mesures`;
    if (log.length) {
      logEl.innerHTML = log.slice().reverse().map(entry => {
        const snap = entry.snap || {};
        const rx = entry.real.x, ry = entry.real.y;
        const errs = {
          odom:     snap.odom   ? Math.sqrt((rx-snap.odom.x)**2   + (ry-snap.odom.y)**2)   : null,
          us:       snap.us     ? Math.sqrt((rx-snap.us.x)**2     + (ry-snap.us.y)**2)     : null,
          vision:   snap.vision ? Math.sqrt((rx-snap.vision.x)**2 + (ry-snap.vision.y)**2) : null,
          fusion:   snap.fusion ? Math.sqrt((rx-snap.fusion.x)**2 + (ry-snap.fusion.y)**2) : null,
        };
        return `<div class="log-entry">
          <div class="log-step">#${entry.step}</div>
          <div class="log-method-vals">
            ${Object.entries(errs).map(([k,e]) => {
              if(e==null) return `<div class="log-err"><span>--</span></div>`;
              const cls = e>10?'bad':e>5?'warn':'';
              return `<div class="log-err ${cls}"><div style="font-size:8px;color:#aaa">${k}</div><span>${e.toFixed(1)}cm</span></div>`;
            }).join('')}
          </div>
          <div style="font-size:9px;color:#888">t=${entry.t.toFixed(1)}s</div>
        </div>`;
      }).join('');
    }

    // Auto-snapshot si session active
    if (s.session.running) {
      await takeSingleSnapshot();
    }

  } catch(e) {}
  setTimeout(pollState, 600);
}

// Init
async function init() {
  await refreshAruco();
  await refreshData();
  pollState();
  // Refresh data recap toutes les 5s
  setInterval(refreshData, 5000);
}

// Keyboard
document.addEventListener('keydown', e => {
  if (document.activeElement.tagName === 'INPUT') return;
  if (e.key === 'ArrowUp')    cmd('AVANCE');
  if (e.key === 'ArrowDown')  cmd('RECUL');
  if (e.key === 'ArrowLeft')  cmd('PIVOT_G');
  if (e.key === 'ArrowRight') cmd('PIVOT_D');
  if (e.key === ' ') { e.preventDefault(); cmd('STOP'); }
});

init();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index(): return HTML

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)