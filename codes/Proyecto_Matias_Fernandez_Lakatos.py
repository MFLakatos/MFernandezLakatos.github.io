### INFORMACIÓN A TENER EN CUENTA: 

# Los videos no deben poseer más de un objeto con igual color. Si bien es posible diferenciar diferentes tonalidades de un color es mejor no llevar al extremo el código.
# Una posible mejora sería combinar este trackeo de color con otro que use la textura del objeto.
# Al ser objetos sencillos no se pueden sacar muchos descriptores y por ende keypoints, por eso se trabaja con el color.
# Un perfeccionamiento para lograr obtener un mejor trackeo de los objetos sería entrenar a un algoritmo con cada objeto en diferentes posiciones y con incidencias de luz variadas, mostrándole cuál es el objeto en cada instante. Deep Learning.


#1# Los videos de las tres cámaras deben presentarse como vid_cam_09 / 10 / 11
#2# Los límites de los colores en el espacio HSV de los objetos fueron obtenidos con el archivo range-detector.py
#3# Los archivos .json tienen la siguiente estructura: '12 dígitos conteniendo el número de la imagen'+ '_rendered_18285509_keypoints' 'nada/_1/_2'.json Ejemplo: '000000000321_rendered_18285509_keypoints.json'
#4# En la terminal se deben poner el siguiente link y adaptarlo a cada sitio donde es arrancado. Abrir la terminal desde la carpeta donde se encuentra el archivo "tracker_objetos_varias_camaras_color-centroide.py":

# python Proyecto_Matias_Fernandez_Lakatos.py --video <link a la carpeta donde estan los videos> --json <link a la carpeta donde estan los archivos .json> --confianza 0.5 --escala 600 --buffer 42 --guardar imgs --objeto verde

# Las opciones para el objeto (o) son: 'rojo' 'amarillo' y 'verde'
# La confianza (c) es el valor que arroja los .json del OpenPose con respecto a una determinada predicción.
# La escala (e) es con el fin de mejorar la visualización para distintas máquinas
# El buffer (b) corresponde a cuán larga es la curva que sigue al centroide, en mi caso, de color roja.
# Opción disponible para guardar (g) o no los videos. "si": guarda.
###

# Paquetes necesarios

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import urllib.request as request
import json

# Llamar a las variables necesarias

ap = argparse.ArgumentParser()
ap.add_argument("-o","--objeto", type=str, required = True, help="objeto a trackear")
ap.add_argument("-v","--video", required = True, help="camino a la carpeta de videos")
ap.add_argument("-j","--json", required = True,help="path to the json's files")
ap.add_argument("-c","--confianza",type=float,default=0.2, help="parámetro de confianza para OpenPose")
ap.add_argument("-e","--escala", type=int, default=600, help="parámetro de re-escaleo")
ap.add_argument("-b","--buffer" , type=int, default=64 , help="max buffer size")
ap.add_argument("-g","--guardar", type=str, required = True, help="guarda video si se coloca la opción 'vid', guarda las imágenes de los frames si la opción es 'imgs'")
args = vars(ap.parse_args())


# Definimos los límites de color, en el espacio de color HSV, para cada objeto
# Lo hacemos con el archivo: "range-detector.py"

if args["objeto"]=='rojo':
	colorLower = (0, 187,0)		#RedLower
	colorUpper = (6, 244,255)	#RedUpper
elif args["objeto"]=='amarillo': #v2
	colorLower = (22, 126,0)	#YellowLower
	colorUpper = (24,154,255)	#YellowUpper
elif args["objeto"]=='verde': #v2
	colorLower = (46,82,0)	#VerdeLower
	colorUpper = (65,142,255)	#VerdeUpper
elif args["objeto"]=='azul':
	colorLower = (92,39,84)		#VerdeLower
	colorUpper = (120,85,167)	#VerdeUpper

# Puntos de la traza de centroides detectados. Uno para cada video.

pts = [ deque(maxlen=args["buffer"]), deque(maxlen=args["buffer"]), deque(maxlen=args["buffer"]) ] 

# Importo los videos de las tres cámaras:

vs = [cv2.VideoCapture(args["video"]+'/vid_cam_09.mp4'), cv2.VideoCapture(args["video"]+'/vid_cam_10.mp4'), cv2.VideoCapture(args["video"]+'/vid_cam_11.mp4')]

# Descriptores de los videos (tomo cam09 como la representativa)

length = int(vs[0].get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(vs[0].get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vs[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = vs[0].get(cv2.CAP_PROP_FPS)
print('cantidad de frames:',length,'largo del video:',width,'ancho del video:' ,height,'fps:',fps)

# Genero los elementos que harán la escala y el video que guardaré

Factor_escala =  args["escala"]
imgScale      =  Factor_escala/width
newX,newY     =  width*imgScale, height*imgScale
if args["guardar"]=='vid':
	fourcc        =  cv2.VideoWriter_fourcc(*'MP4V')   #esto sirve para que salga un video, pero más lento. Palabra clave: videoout
	out           =  [ 
cv2.VideoWriter(args["video"]+'/outputcam09_e-'+repr(args["escala"])+ '_b-'+repr(args["buffer"])+ '_c-0'+repr(int(10*args["confianza"]))+ '_o-'+args["objeto"]+ '.mp4',fourcc, 20.0, (int(newX),int(newY))), 
cv2.VideoWriter(args["video"]+'/outputcam10_e-'+repr(args["escala"])+ '_b-'+repr(args["buffer"])+ '_c-0'+repr(int(10*args["confianza"]))+ '_o-'+args["objeto"]+ '.mp4',fourcc, 20.0, (int(newX),int(newY))), 
cv2.VideoWriter(args["video"]+'/outputcam11_e-'+repr(args["escala"])+ '_b-'+repr(args["buffer"])+ '_c-0'+repr(int(10*args["confianza"]))+ '_o-'+args["objeto"]+ '.mp4', fourcc, 20.0, (int(newX),int(newY)))]    #esto sirve para que salga un video, pero más lento. Palabra clave: videoout


#iniciar conteos

tiempo_R_toca = np.zeros((length+1,3))
tiempo_L_toca = np.zeros((length+1,3))
cambia = 'si'			# Utilizo esta variable a modo de switch para graficar o no los puntos de OpenPose, la prendo al grabar nueva info en niño, la pago al finalizar el loop
restriccion_distancia = 8  	# Para distancias menores a ésta considero que el bebé toca el objeto.

# Loop para las tres cámaras
for j in range(3):
	counter_frames = 0
	while True:
		# Toma el frame del video
		oriframe= vs[j].read()
		oriframe = oriframe[1]
		
		# Si es el final del video, que rompa el while
		if oriframe is None:
			break

		# Sumamos un frame más ya que continuamos en el loop
		counter_frames += 1

		# (resize the frame)
		frame = cv2.resize(oriframe,(int(newX),int(newY)))
		# Borroneamos para hacer un pasa alto en las frecuencias: blur
		blurred = cv2.GaussianBlur(frame, (11, 11), 0)
		# Pasamos al espacio de color HSV
		hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

		# Construimos una máscara con los límites impuestos en el espacio de color
		mask= cv2.inRange(hsv, colorLower, colorUpper)

		# a series of dilations and erosions to remove any small
		# blobs left in the mask
# Estas dos líneas son análogas a cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel).
# Mediante dilataciones y erosiones removemos las pequeñas burbujas que quedan en la máscara
# Aquí se presenta la solución al problema de los puntos falsos positivos de la máscara de color que es distinto para cada objeto.
# Es más artesanal este paso, y adaptado a cada objeto.

		if args["objeto"]   == 'rojo':
			mask = cv2.erode(mask,  None, iterations=3)	 
			mask = cv2.dilate(mask, None, iterations=3)	
		elif args["objeto"] == 'amarillo' or args["objeto"] == 'verde':
			mask = cv2.erode(mask,  None, iterations=3)	
			mask = cv2.dilate(mask, None, iterations=3)	 
		#Para visualizar lo que ve el algorítmo como máscara:
		cv2.imshow("Mask", mask)
	
		# Encuentra los contornos de la máscara
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		# SIMPLE (guarda menos elementos, más rápida) / NONE	
		# Toma la versión correcta de OpenCV
		cnts = imutils.grab_contours(cnts)
		center = None
		radius = None
		rect   = None

		# only proceed if at least one contour was found
		if len(cnts) > 0: 
			# Encuentra el controno más grande de la máscara y lo utiliza para encontrar la mínima figura correspondiente al objeto y su respectivo centroide. Por ejemplo, un rectángulo o un círculo
			c = max(cnts, key=cv2.contourArea)
			# Sabemos la figura que se adapta a cada objeto
			if args["objeto"]=='rojo':
				rect = cv2.minAreaRect(c)
			elif args["objeto"]=='amarillo' or args["objeto"]=='verde':
				((x, y), radius) = cv2.minEnclosingCircle(c)	
				
			# moments trae el centroide:
			# cx = int(M['m10']/M['m00']) & cy = int(M['m01']/M['m00'])
			M = cv2.moments(c) 
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
			# Dibujo el centroide de color Rojo (0,0,255) [BGR]
			cv2.circle(frame, center, int(10*imgScale), (0, 0, 255), -1) 

			# Procede si radio mayor a un valor,
			if radius is not None:
				if radius > 10*imgScale:
					# Dibuja un círculo amarillo (0, 255,255) centrado en el centroide
					cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255,255), 2)
			# o altura AND ancho mayor a un valor.
			if rect is not None:
				if rect[1][0]>10*imgScale and rect[1][1]>10*imgScale:
					box = cv2.boxPoints(rect)
					box = np.int0(box)
					# Dibuja un rect amarillo (0, 255,255) centrado en el centroide
					cv2.drawContours(frame,[box],0,(0,255,255),2) 
					
		
			# Actualiza los puntos de trackeo (queue). Lo hace tal que los va colocando a la izquierda, por eso la definción de thickness más adelante.
			pts[j].appendleft(center)

		# Loop en los puntos de trackeo. Acá hago el haz que se va desintegrando
		for i in range(1, len(pts[j])):
			
			# Si alguno de los dos últimos en None, los ignoro
			if pts[j][i - 1] is None or pts[j][i] is None:
				continue
	
			# Si no, defino el ancho de la línea que conecta los puntos de forma tal que
			# cuanto más "viejos" sean los puntos más se achiquen. 
			thickness = 1+int(np.sqrt(args["buffer"] / float(i + 1)) * 2*imgScale)
			cv2.line(frame, tuple(pts[j][i-1]),tuple(pts[j][i]), (0,0,255), thickness)
			

		##################	JASON FILES	##################

		# Archivos: 000000000 número de archivo _rendered_18285509_keypoints  _1 _2 .json	
		
		# Hay 12 dígitos de números, pongo el número de la imagen y lleno de ceros hasta completar los 12 dígitos		
		num_archivo = repr(counter_frames-1).zfill(12)
		if j==0:
			with open(args["json"]+'/'+num_archivo+'_rendered_18285509_keypoints.json') as f:
				data = json.load(f)
		elif j==1:
			with open(args["json"]+'/'+num_archivo+'_rendered_18285509_keypoints_1.json') as f:
				data = json.load(f)
		else:
			with open(args["json"]+'/'+num_archivo+'_rendered_18285509_keypoints_2.json') as f:
				data = json.load(f)

		# Extraigo valores del json:
		persona_1 = data['people'][0]
		pose_keypoints_2d_1 = persona_1["pose_keypoints_2d"]	
		pos_cabeza = 1 # posición en el archivo .json			
		xc1 = pose_keypoints_2d_1[int((pos_cabeza-1)*3)]
		yc1 = pose_keypoints_2d_1[int((pos_cabeza-1)*3+1)]
		# Prueba ver si hay datos de otro esqueleto y compara con el que ya está para ver si el segundo es el niño o es simplemente el primero
		try:
			persona_2 = data['people'][1]
			pose_keypoints_2d_2 = persona_2["pose_keypoints_2d"]
			xc2 = pose_keypoints_2d_2[int((pos_cabeza-1)*3)]
			yc2 = pose_keypoints_2d_2[int((pos_cabeza-1)*3+1)] 
			# En la cámara 10, la opuesta a la puerta, el niño tiene la condición de que está a una y menor, no una x mayor. 
			if (xc1-xc2<0) and j != 1: 	# Si xc1 < xc2 -> me quedo con xc2 (niño derecha)
				niño   = data['people'][1]
			elif (yc1-yc2<0) and j == 1: 	# Si yc1 < yc2 -> me quedo con yc2 (niño abajo)
				niño   = data['people'][1]
		except:
			niño   = data['people'][0]
		
		pose_keypoints_2d = niño["pose_keypoints_2d"]	
		hand_left  	  = niño["hand_left_keypoints_2d"]
		hand_right 	  = niño["hand_right_keypoints_2d"]
		if cambia == 'si':
			# Cabeza:
			xc = pose_keypoints_2d[int((pos_cabeza-1)*3)]
			yc = pose_keypoints_2d[int((pos_cabeza-1)*3+1)] 
			cc = pose_keypoints_2d[int((pos_cabeza-1)*3+2)]
			cabeza_pos = (int(xc*imgScale),int(yc*imgScale))
			cv2.circle(frame, cabeza_pos, int(8*imgScale), (255,255,255), -1)	#Blanco

		# Pulgar de la mano derecha:
		pos_pulgar_RHand = 4 # posición en el archivo .json	
		xpRH = hand_right[int((pos_pulgar_RHand-1)*3)]
		ypRH = hand_right[int((pos_pulgar_RHand-1)*3+1)] 
		cpRH = hand_right[int((pos_pulgar_RHand-1)*3+2)]
		# Junto con la imagen, las posiciones se ajustan a la nueva escala.
		#Factor_escala = 600
		#imgScale = Factor_escala/width
		RHand_pulgar_pos = (int(xpRH*imgScale),int(ypRH*imgScale))
		cv2.circle(frame, RHand_pulgar_pos, int(8*imgScale), (255,51,51), -1) 		#azul claro

		# Pulgar de la mano izquierda:
		pos_pulgar_LHand = 4	
		xpLH = hand_left[int((pos_pulgar_LHand-1)*3)]
		ypLH = hand_left[int((pos_pulgar_LHand-1)*3+1)] 
		cpLH = hand_left[int((pos_pulgar_LHand-1)*3+2)]
		LHand_pulgar_pos = (int(xpLH*imgScale),int(ypLH*imgScale))
		cv2.circle(frame, LHand_pulgar_pos, int(8*imgScale), (0 ,204, 0), -1) 		#verde claro

		# Muñeca derecha:
		pos_RWrist = 5	
		xRW = pose_keypoints_2d[int((pos_RWrist-1)*3)]
		yRW = pose_keypoints_2d[int((pos_RWrist-1)*3+1)] 
		cRW = pose_keypoints_2d[int((pos_RWrist-1)*3+2)]
		RWrist_pos = (int(xRW*imgScale),int(yRW*imgScale))
		cv2.circle(frame, RWrist_pos, int(8*imgScale), (255,255,51), -1) 		#celeste

		# Muñeca izquierda:
		pos_LWrist = 8 
		xLW = pose_keypoints_2d[int((pos_LWrist-1)*3)]
		yLW = pose_keypoints_2d[int((pos_LWrist-1)*3+1)] 
		cLW = pose_keypoints_2d[int((pos_LWrist-1)*3+2)]
		LWrist_pos = (int(xLW*imgScale),int(yLW*imgScale))
		cv2.circle(frame, LWrist_pos, int(8*imgScale), (102,255,178), -1) 		#verde agua

		##########################################################

		# Defino la distancia de forma tal que si estan los objetos lo más alejado posible da 100.
		if center is not None and center != (0,0):

			if cRW > args["confianza"]:
				cambia == 'si'
				# Distancia objeto-muñeca derecha
				dist_obj_Rwrist = 100*np.sqrt((RWrist_pos[0]-center[0])**2 + (RWrist_pos[1]-center[1])**2 )/(np.sqrt(newX**2+newY**2))
				cv2.line(frame, RWrist_pos,(int(center[0]),int(center[1])), (255,0,0), int(4*imgScale))  #En azul

				# Guardo los puntos potenciales al contacto bebé-objeto:
				if dist_obj_Rwrist < restriccion_distancia:
					tiempo_R_toca[ counter_frames][j] = counter_frames/fps
					cv2.putText(frame, "TOCA", (int(newX*0.02),  int(newY*0.8)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

				info_R = ["Dist MD-O", dist_obj_Rwrist]
				text_R = "{}: {}".format(info_R[0],info_R[1])
				cv2.putText(frame, text_R, (int(newX*0.02),  int(newY*0.9)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

			if cLW > args["confianza"]:
				cambia == 'si'
				# Distancia objeto-muñeca izquierda
				dist_obj_Lwrist = 100*np.sqrt((LWrist_pos[0]-center[0])**2 + (LWrist_pos[1]-center[1])**2 )/(np.sqrt(newX**2+newY**2))
				cv2.line(frame, LWrist_pos,(int(center[0]),int(center[1])), (0,255,0), int(4*imgScale))  #En verde 

				if dist_obj_Lwrist < restriccion_distancia:
					tiempo_L_toca[counter_frames][j] = counter_frames/fps

				info_L = ["Dist MI-O", dist_obj_Lwrist]
				text_L = "{}: {}".format(info_L[0],info_L[1])
				cv2.putText(frame, text_L, (int(newX*0.02),  int(newY*0.95)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

			else:
				cambia == 'no'

		# Muestra el video y la máscara.
		cv2.imshow("Frame", frame)

		# Exporto el frame si la opción guardar es 'imgs'
		if args["guardar"]=='imgs':
			outfile = args["video"]+'/imagenes_'+args["objeto"]+'/outimg_cam'+repr(j)+'_'+ repr(counter_frames).zfill(12)+'.png'
			cv2.imwrite(outfile, frame)
			outmask = args["video"]+'/imagenes_'+args["objeto"]+'/outmask_cam'+repr(j)+'_'+ repr(counter_frames).zfill(12)+'.png'
			cv2.imwrite(outmask, mask)

		# Escribo el video de salida si se pone la opción 'vid'
		if args["guardar"]=='vid':
			out[j].write(frame) #esto sirve para que salga un video, pero más lento. Palabra clave: videoout

		cv2.moveWindow('Frame',  0  ,0) # x horizontal(izq-der), y vertical(arr-aba)
		cv2.moveWindow('Mask' , 700 ,0) 
		key = cv2.waitKey(1) & 0xFF

		# Si la tecla 'q' es presionada, termina el loop de esta cámara
		if key == ord("q"):
			print('cantidad de frames hasta ahora:',counter_frames)
			break

vs[2].release()
# Cierra todas las ventanas
cv2.destroyAllWindows()

# Supongo que el video no empieza con el bebé tocando un objeto
# tiempo_RL_toca es una variable auxiliar para simplificar código, el for que recorre las 'k' es para tomar tanto la muñeca derecha como la izquierda en un sólo pedazo de código.
tiempo_RL_toca= [tiempo_R_toca,tiempo_L_toca]

# contacto[0] será para la muñeca derecha, contacto[1] para la izquierda
contacto = [[],[]]
for k in range(2):
	tiempo_contacto = tiempo_RL_toca[k]
	for i in range(int(length)):
		# Todos tendrán la misma información, el tema es ver si son diferentes a cero
		# A continuación vemos si dos para al menos dos cámaras el bebé toca el objeto	
		a, b, c     = tiempo_RL_toca[k][i-1,0], tiempo_RL_toca[k][i-1,1], tiempo_RL_toca[k][i-1,2] #pasado
		#print(a,b,c)
		aa, bb, cc  = tiempo_RL_toca[k][i+0,0], tiempo_RL_toca[k][i+0,1], tiempo_RL_toca[k][i+0,2] #presente
		aaa,bbb,ccc = tiempo_RL_toca[k][i+1,0], tiempo_RL_toca[k][i+1,1], tiempo_RL_toca[k][i+1,2] #futuro
		# Si la cámara 'a' y la cámara 'b' detecta contacto, entonces...
		if (aa*bb != 0):
			# Si en el paso anterior no hubo contacto para UNA o NINGUNA CÁMARA entonces estamos en el inicio del contacto (un contacto no es considerado contacto)
			if  (sum([a==0,b==0,c==0])>=2):
				inicio = aa # puede ser bb también.
			# Si en el próximo ínidice termina el video, entonces 
			if i+1 == int(length):
				# Si no detecta ningún contacto en alguna cámara, se termina ya
				if (aaa+bbb+ccc == 0):
					final = aa
				# De lo contrario, termina el video tocando el objeto				
				else:
					if (aaa !=0):
						final = aaa
					elif (bbb !=0):
						final = bbb
					else:
						final = ccc				 
				contacto[k].append([inicio,final,final-inicio])
				break
			elif (sum([aaa==0,bbb==0,ccc==0])>=2):
				final = aa
				contacto[k].append([inicio,final,final-inicio])
		elif (aa*cc != 0):
			# Si en el paso anterior no hubo contacto en NINGUNA CÁMARA entonces estamos en el inicio del contacto
			if  (sum([a==0,b==0,c==0])>=2):
				inicio = aa # puede ser bb también.
			# Si en el próximo ínidice termina el video, entonces 
			if i+1 == int(length):
				# Si no detecta ningún contacto en alguna cámara, se termina ya
				if (aaa+bbb+ccc == 0):
					final = aa
				# De lo contrario, termina el video tocando el objeto				
				else:
					if (aaa !=0):
						final = aaa
					elif (bbb !=0):
						final = bbb
					else:
						final = ccc
				contacto[k].append([inicio,final,final-inicio])
				break
			elif (sum([aaa==0,bbb==0,ccc==0])>=2):
				final = aa
				contacto[k].append([inicio,final,final-inicio])
		elif (bb*cc != 0):
			# Si en el paso anterior no hubo contacto en NINGUNA CÁMARA entonces estamos en el inicio del contacto
			if  (sum([a==0,b==0,c==0])>=2):
				inicio = bb # puede ser bb también.
			# Si en el próximo ínidice termina el video, entonces 
			if i+1 == int(length):
				# Si no detecta ningún contacto en alguna cámara, se termina ya
				if (aaa+bbb+ccc == 0):
					final = bb
				# De lo contrario, termina el video tocando el objeto				
				else:
					if (aaa !=0):
						final = aaa
					elif (bbb !=0):
						final = bbb
					else:
						final = ccc
				contacto[k].append([inicio,final,final-inicio])
				break
			elif (sum([aaa==0,bbb==0,ccc==0])>=2):
				final = bb
				contacto[k].append([inicio,final,final-inicio])

# Si el final de un contacto y el inicio del siguiente se diferencian en un 'intervalo_t' entonces uno los dos segmentos.
intervalo_t = 0.3 #segundos
contacto_mod = [ [] , [] ]
for k in range(2):
	contacto[k].append([length/fps+10,length/fps+10,length/fps+10]) #Esto lo hago por un problema de programación que no encuentro. De no hacerlo no me toma los últimos puntos cuando junto los tiempos que distan menos de intervalo_t
	B = np.zeros(len(contacto[k])-1)
	for i in range(len(contacto[k])-1):
		B[i] = contacto[k][i+1][0]-contacto[k][i][1] < intervalo_t #Relaiso la resta entre el inicio del frame i+1 con el final del frame i
	i = 0
	inicio = contacto[k][0][0]
	final  = contacto[k][0][1]
	if len(contacto[k])==2:
		contacto_mod[k].append([inicio,final,final-inicio]) #Si hay dos veces que toca, una será el elemento agregado: 
	else:
		while i < len(B):
			i+=1
			if B[i-1]==1:
				final =  contacto[k][i][1] #Voy a tomar el siguiente por si es falso B
			else:
				contacto_mod[k].append([inicio,final,final-inicio])
				inicio = contacto[k][i][0] #Voy a tomar el siguiente por si es falso B
				final =  contacto[k][i][1] #Voy a tomar el siguiente por si es falso B


# Imprimo los valores en la terminal
print('Para la mano derecha:')
if len(contacto_mod[0])==0:
	print('No toca la mano derecha el objeto')
for i in range(len(contacto_mod[0])):
	print('inicio:',contacto_mod[0][i][0], 'final:', contacto_mod[0][i][1], 'duración:', contacto_mod[0][i][2])

print('Para la mano izquierda:')
if len(contacto_mod[1])==0:
	print('No toca la mano izquierda el objeto')
for i in range(len(contacto_mod[1])):
	print('inicio:',contacto_mod[1][i][0], 'final:', contacto_mod[1][i][1], 'duración:', contacto_mod[1][i][2])

# Genero el archivo .dat
# Guardo los inicios, finales y duraciones
header = "Para el objeto "+args["objeto"]+", con escala " +repr(args["escala"])+", buffer "+repr(args["buffer"])+" y confianza 0."+repr(int(10*args["confianza"]))+":"

f = open('data_objeto_'+args["objeto"]+'.dat', 'wb')
np.savetxt(f, [], header=header) 

# Defino los textos a colocar en el archivo
string  = ["Para la distancia objeto - muñeca derecha:"]
string2 = ["Para la distancia objeto - muñeca izquierda:"]
infidu  = ["inicio,final,duración"]
espacio = [" "]

np.savetxt(f,espacio,fmt="%s")
np.savetxt(f,string,fmt="%s")
np.savetxt(f,infidu,fmt="%s")
for i in range(len(contacto_mod[0])):
    data = np.column_stack((contacto_mod[0][i][0], contacto_mod[0][i][1],contacto_mod[0][i][2]))
    np.savetxt(f, data,delimiter='  ',fmt='%1.3f')

np.savetxt(f,espacio,fmt="%s")
np.savetxt(f,string2,fmt="%s")
np.savetxt(f,infidu,fmt="%s")
for i in range(len(contacto_mod[1])):
    data = np.column_stack((contacto_mod[1][i][0], contacto_mod[1][i][1],contacto_mod[1][i][2]))
    np.savetxt(f, data,delimiter='  ',fmt='%1.3f')

f.close()
