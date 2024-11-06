from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import mediapipe as mp

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

####### CONFIGURACIÓN PERSONALIZADA ########

titulo = "ITCELAYA"

# Calibración Mascara 1 (0 - 255) - FORMATO HSV

Hmin_M1 = 40
Smin_M1 = 100
Vmin_M1 = 100

Hmax_M1 = 80
Smax_M1 = 255
Vmax_M1 = 255

# Calibración Mascara 2 (0 - 255)   - FORMATO HSV

Hmin_M2 = 160
Smin_M2 = 100
Vmin_M2 = 100

Hmax_M2 = 180
Smax_M2 = 255
Vmax_M2 = 255

def principal():

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if ret:

        # DEFINIR ESPACIO DE TRABAJO
        x,y,w,h = espacio_trabajo(frame)

        # Aplicacion de mascaras
        mask1,mask2 = mascaras(frame)

        # Detectar objeto

        objeto1 = "Rojo"
        color1 = (255,0,10)   #RGB

        objeto2 = "Verde"
        color2 = (255,255,0)   #RGB

        detectar_objeto(frame,mask1,color1,objeto1,x,y,w,h)
        detectar_objeto(frame, mask2, color2, objeto2,x,y,w,h)

        # Detección de manos
        detectar_manos(frame,x,y,w,h)

        #Interfaz
        tkinter_videoStream(frame,mask1,mask2)

def tkinter_videoStream(frame,mask1,mask2):

    height, width, _ = frame.shape
    frame = cv2.resize(frame, (int(width * 1.32), int(height * 1.32)))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    tkimage = ImageTk.PhotoImage(img)
    video_stream.configure(image=tkimage)
    video_stream.image = tkimage

    mask_color1 = cv2.resize(mask1, (0, 0), fx=0.45, fy=0.45)
    img1 = Image.fromarray(mask_color1)
    tkimage1 = ImageTk.PhotoImage(img1)
    video_color1.configure(image=tkimage1)
    video_color1.image = tkimage1

    mask_color2 = cv2.resize(mask2, (0, 0), fx=0.45, fy=0.45)
    img2 = Image.fromarray(mask_color2)
    tkimage2 = ImageTk.PhotoImage(img2)
    video_color2.configure(image=tkimage2)
    video_color2.image = tkimage2

    ventana.after(1, principal)

def detectar_manos(frame,x,y,w,h):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    color = (0, 177, 46)  # Verde

    # Procesar si hay landmarks detectados
    if results.multi_hand_landmarks is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x_h = int(landmark.x * frame.shape[1])
                y_h = int(landmark.y * frame.shape[0])

                # Comprobar si los puntos de la mano están dentro del rectángulo
                if x <= x_h <= (x+w) and y <= y_h <= (y+h):
                    color = (0, 0, 255)  # Cambiar el color del rectángulo a rojo si al menos un punto está dentro
                    cv2.rectangle(frame, (int(x+(w/2)-50), int(y+(h/2)-20)), (int(x+(w/2)+50), int(y+(h/2)+15)), (0,255,255), -1)
                    cv2.putText(frame, "PELIGRO", (x +int(w/2)-45, y + int(h/2)+5), cv2.FONT_HERSHEY_DUPLEX, 0.7,(0, 0, 0), 1, cv2.LINE_AA)

                    break  # Si un punto está dentro, no necesitamos seguir comprobando

                 # Dibujar los landmarks en la mano
                mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 100, 255), thickness=3, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 100, 0), thickness=4, circle_radius=3))


    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(frame, (x, y), (x + 160, y + 25), color, -1)
    cv2.putText(frame, "Espacio de trabajo", (x+5,y+17), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1,cv2.LINE_AA)

def espacio_trabajo(frame):

    height, width, _ = frame.shape

    # Aplicación de mascara color 1
    x = int(Ubi_x.get()*(width/100))  # Coordenada x del área de detección
    y = int(Ubi_y.get()*(height/100))  # Coordenada y del área de detección
    w = int((Ancho_area.get()*(width/70))+(width*0.30))  # Ancho del área de detección
    h = int((Altura_area.get()*(width/70))+(height*0.30))  # Altura del área de detección

    return x,y,w,h

def mascaras(frame):

    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Aplicación de mascara color 1
    mask1_min = np.array([int(Hmin.get()), int(Smin.get()), int(Vmin.get())], np.uint8)
    mask2_max = np.array([int(Hmax.get()), int(Smax.get()), int(Vmax.get())], np.uint8)

    mask1 = cv2.inRange(frameHSV, mask1_min, mask2_max)

    # Aplicación de mascara color 2

    mask2_min = np.array([int(Hmin1.get()), int(Smin1.get()), int(Vmin1.get())], np.uint8)
    mask2_max = np.array([int(Hmax1.get()), int(Smax1.get()), int(Vmax1.get())], np.uint8)

    mask2 = cv2.inRange(frameHSV, mask2_min, mask2_max)

    return mask1,mask2

def detectar_objeto(frame,mask,color,texto,x,y,w,h):

    # Encuentra los contornos en la máscara
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Recorre cada contorno
    for c in contornos:
        area = cv2.contourArea(c)
        if area > 1000:
            # Verifica si todos los puntos del contorno están dentro del área de detección
            is_inside = True  # Suponemos que el contorno está dentro

            for point in c:  # `c` es una lista de puntos del contorno
                px, py = point[0]  # Coordenadas x, y de cada punto
                if not (x <= px <= x + w and y <= py <= y + h):
                    is_inside = False  # Si un punto está fuera, marcamos como fuera
                    break  # No es necesario seguir verificando

            if is_inside:
                # Si el contorno está dentro del área de detección, calcular el centroide
                M = cv2.moments(c)
                if M["m00"] == 0:
                    M["m00"] = 1
                cx = int(M["m10"] / M["m00"])  # Centroide en x
                cy = int(M["m01"] / M["m00"])  # Centroide en y
                cv2.circle(frame, (cx, cy), 7, color, -1)  # Dibuja el centroide
                cv2.putText(frame, f"{texto}", (cx - 10, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,cv2.LINE_AA)
                # Dibuja el contorno usando la envolvente convexa
                nuevoContorno = cv2.convexHull(c)
                cv2.drawContours(frame, [nuevoContorno], 0, color, 3)


####################### CONFIGURACIONES DE LA INTERFAZ #############################

ventana = Tk()
ventana.title(titulo)

# Tamaño de la ventana de la computadora

ancho_pantalla = int(ventana.winfo_screenwidth())  # 1920
altura_pantalla = int(ventana.winfo_screenheight())  # 1080
ventana.geometry(f"{int(ancho_pantalla)}x{int(altura_pantalla)}+0+0")
ventana.config(bg="gray1")

# Panel camara

panel_camara = Frame(ventana, relief=FLAT,bg="gray1")
panel_camara.place(x=int(ancho_pantalla*0.01), y=int(altura_pantalla*0.1))
video_stream = Label(panel_camara,bg="gray15",bd=10)
video_stream.grid(row=0)

# Etiqueta titulo
etiqueta_titulo2 = Label(ventana, text='Calibración de Máscaras', fg='snow', font=("Franklin Gothic Demi", 20),bg="gray15", width=45)
etiqueta_titulo2.place(x=ancho_pantalla*0.58, y=altura_pantalla*0.1)

# Etiqueta titulo
etiqueta_titulo3 = Label(ventana, text='Área de Detección', fg='snow', font=("Franklin Gothic Demi", 20),bg="gray15", width=45)
etiqueta_titulo3.place(x=ancho_pantalla*0.58, y=altura_pantalla*0.69)

# Panel de configuración

confifuracion_mask1 = Frame(ventana,relief=FLAT,width=300, height=100,bg="gray1")
confifuracion_mask1.place(x=ancho_pantalla*0.78, y=altura_pantalla*0.18)

confifuracion_mask2 = Frame(ventana,relief=FLAT,width=300, height=100,bg="gray1")
confifuracion_mask2.place(x=ancho_pantalla*0.78, y=altura_pantalla*0.44)

frame_mascaras = Frame(ventana,relief=FLAT,bg="gray1")
frame_mascaras.place(x=ancho_pantalla*0.58, y=altura_pantalla*0.16)

configuracion_area = Frame(ventana,relief=FLAT,width=300, height=100,bg="gray1")
configuracion_area.place(x=ancho_pantalla*0.58, y=altura_pantalla*0.75)

video_color1 = Label(frame_mascaras,bg="floral white")
video_color1.grid(row=0,column=0)

video_color2 = Label(frame_mascaras,bg="floral white")
video_color2.grid(row=1,column=0,pady=10)

# Etiqueta titulo
etiqueta_titulo = Label(ventana, text=titulo, fg='snow', font=("Franklin Gothic Demi", 35),bg="forest green", width=70)
etiqueta_titulo.place(relx=0.5, rely=0.04, anchor=CENTER)

# Inicializa MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.7)

###################### CONFIGURACION DE SLIDERS #############################

def RangoHSV(int):
    Hmin.set(sliderHmin.get())
    Hmax.set(sliderHmax.get())
    Smin.set(sliderSmin.get())
    Smax.set(sliderSmax.get())
    Vmin.set(sliderVmin.get())
    Vmax.set(sliderVmax.get())

def RangoHSV1(int):
    Hmin1.set(sliderHmin1.get())
    Hmax1.set(sliderHmax1.get())
    Smin1.set(sliderSmin1.get())
    Smax1.set(sliderSmax1.get())
    Vmin1.set(sliderVmin1.get())
    Vmax1.set(sliderVmax1.get())

def Area(int):
    Ancho_area.set(sliderAncho.get())
    Altura_area.set(sliderAltura.get())
    Ubi_x.set(sliderUbi_x.get())
    Ubi_y.set(sliderUbi_y.get())


# Rango del 1er color
Hmin = IntVar()
Hmax = IntVar()
Smin = IntVar()
Smax = IntVar()
Vmin = IntVar()
Vmax = IntVar()

# Rango del 2do color
Hmin1 = IntVar()
Hmax1 = IntVar()
Smin1 = IntVar()
Smax1 = IntVar()
Vmin1 = IntVar()
Vmax1 = IntVar()

# Area
Ancho_area = IntVar()
Altura_area = IntVar()
Ubi_x = IntVar()
Ubi_y = IntVar()

# Slinders para detección de area

sliderUbi_x = Scale(configuracion_area,label = 'Eje x', from_=0, to=100, orient=HORIZONTAL,command=Area,length=300,bg="cyan4",fg='black',font=("Franklin Gothic Demi", 10))
sliderUbi_x.grid(row=0,column=0,padx=10)
sliderUbi_x.set(0)

sliderUbi_y = Scale(configuracion_area,label = 'Eje y', from_=0, to=100, orient=HORIZONTAL,command=Area,length=300,bg="cyan4",fg='black',font=("Franklin Gothic Demi", 10))
sliderUbi_y.grid(row=1,column=0)
sliderUbi_y.set(0)

sliderAncho = Scale(configuracion_area,label = 'Anchura', from_=0, to=70, orient=HORIZONTAL,command=Area,length=300,bg="cyan4",fg='black',font=("Franklin Gothic Demi", 10))
sliderAncho.grid(row=0,column=1)
sliderAncho.set(0)

sliderAltura = Scale(configuracion_area,label = 'Altura', from_=0, to=70, orient=HORIZONTAL,command=Area,length=300,bg="cyan4",fg='black',font=("Franklin Gothic Demi", 10))
sliderAltura.grid(row=1,column=1)
sliderAltura.set(0)

# Slinders para el 1er color

sliderHmin = Scale(confifuracion_mask1,label = 'Hmin (Color)', from_=0, to=179, orient=HORIZONTAL,command=RangoHSV,length=150,bg="skyBlue4",fg='black',font=("Franklin Gothic Demi", 10))
sliderHmin.grid(row=0,column=0)
sliderHmin.set(int(Hmin_M1))

sliderSmin = Scale(confifuracion_mask1,label = 'Smin (Saturación)', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV,length=150,bg="skyBlue4",fg='black',font=("Franklin Gothic Demi", 10))
sliderSmin.grid(row=1,column=0)
sliderSmin.set(int(Smin_M1))

sliderVmin = Scale(confifuracion_mask1,label = 'Vmin (Brillo)', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV,length=150,bg="skyBlue4",fg='black',font=("Franklin Gothic Demi", 10))
sliderVmin.grid(row=2,column=0)
sliderVmin.set(int(Vmin_M1))

sliderHmax = Scale(confifuracion_mask1,label = 'Hmax (Color)', from_=0, to=179, orient=HORIZONTAL,command=RangoHSV,length=150,bg="skyBlue4",fg='black',font=("Franklin Gothic Demi", 10))
sliderHmax.grid(row=0,column=1)
sliderHmax.set(int(Hmax_M1))

sliderSmax = Scale(confifuracion_mask1,label = 'Smax (Saturación)', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV,length=150,bg="skyBlue4",fg='black',font=("Franklin Gothic Demi", 10))
sliderSmax.grid(row=1,column=1)
sliderSmax.set(int(Smax_M1))

sliderVmax = Scale(confifuracion_mask1,label = 'Vmax (Brillo)', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV,length=150,bg="skyBlue4",fg='black',font=("Franklin Gothic Demi", 10))
sliderVmax.grid(row=2,column=1)
sliderVmax.set(int(Vmax_M1))

# Slinders para el 2do color

sliderHmin1 = Scale(confifuracion_mask2,label = 'Hmin (Color)', from_=0, to=179, orient=HORIZONTAL,command=RangoHSV1,length=150,bg="Salmon",fg='black',font=("Franklin Gothic Demi", 10))
sliderHmin1.grid(row=3,column=0)
sliderHmin1.set(int(Hmin_M2))

sliderSmin1 = Scale(confifuracion_mask2,label = 'Smin (Saturacion)', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV1,length=150,bg="Salmon",fg='black',font=("Franklin Gothic Demi", 10))
sliderSmin1.grid(row=4,column=0)
sliderSmin1.set(int(Smin_M2))

sliderVmin1 = Scale(confifuracion_mask2,label = 'Vmin (Saturación)', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV1,length=150,bg="Salmon",fg='black',font=("Franklin Gothic Demi", 10))
sliderVmin1.grid(row=5,column=0)
sliderVmin1.set(int(Vmin_M2))

sliderHmax1 = Scale(confifuracion_mask2,label = 'Hmax (Color)', from_=0, to=179, orient=HORIZONTAL,command=RangoHSV1,length=150,bg="Salmon",fg='black',font=("Franklin Gothic Demi", 10))
sliderHmax1.grid(row=3,column=1)
sliderHmax1.set(int(Hmax_M2))

sliderSmax1 = Scale(confifuracion_mask2,label = 'Smax (Saturación)', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV1,length=150,bg="Salmon",fg='black',font=("Franklin Gothic Demi", 10))
sliderSmax1.grid(row=4,column=1)
sliderSmax1.set(int(Smax_M1))

sliderVmax1 = Scale(confifuracion_mask2,label = 'Vmax (Brillo)', from_=0, to=255, orient=HORIZONTAL,command=RangoHSV1,length=150,bg="Salmon",fg='black',font=("Franklin Gothic Demi", 10))
sliderVmax1.grid(row=5,column=1)
sliderVmax1.set(int(Vmax_M2))

ventana.after(1, principal)
ventana.mainloop()
