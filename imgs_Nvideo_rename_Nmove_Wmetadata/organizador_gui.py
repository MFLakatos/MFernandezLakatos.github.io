import os
import tkinter as tk
from tkinter import filedialog, messagebox
import requests
import zipfile
import shutil
from tqdm import tqdm
from media_organizer import MediaOrganizer  # Asegúrate de tener esta clase en el mismo directorio o accesible

APP_DIR = r"C:\ProgramData\MediaOrganizer"
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_ZIP = os.path.join(APP_DIR, "ffmpeg.zip")
FFMPEG_EXE = os.path.join(APP_DIR, "ffmpeg.exe")
FFPROBE_EXE = os.path.join(APP_DIR, "ffprobe.exe")
REGISTRO_JSON = os.path.join(APP_DIR, "procesados.json")


def asegurar_ffmpeg():
    os.makedirs(APP_DIR, exist_ok=True)
    if os.path.exists(FFMPEG_EXE) and os.path.exists(FFPROBE_EXE):
        return

    print("Descargando FFmpeg...")
    response = requests.get(FFMPEG_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(FFMPEG_ZIP, 'wb') as f, tqdm(
        desc="Descargando",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(1024):
            f.write(data)
            bar.update(len(data))

    print("Extrayendo FFmpeg...")
    with zipfile.ZipFile(FFMPEG_ZIP, 'r') as zip_ref:
        temp_dir = os.path.join(APP_DIR, "_temp")
        zip_ref.extractall(temp_dir)
        bin_path = next((os.path.join(root, f)
                         for root, dirs, files in os.walk(temp_dir)
                         for f in files if f == "ffmpeg.exe"), None)
        if bin_path:
            shutil.copy(bin_path, FFMPEG_EXE)
            shutil.copy(bin_path.replace("ffmpeg.exe", "ffprobe.exe"), FFPROBE_EXE)
        shutil.rmtree(temp_dir)
    os.remove(FFMPEG_ZIP)
    print("FFmpeg listo en:", APP_DIR)


def iniciar_gui():
    def seleccionar_origen():
        path = filedialog.askdirectory(title="Seleccionar carpeta de origen")
        if path:
            entrada_origen.set(path)

    def seleccionar_destino():
        path = filedialog.askdirectory(title="Seleccionar carpeta de destino")
        if path:
            entrada_destino.set(path)

    def ejecutar_organizador():
        origen = entrada_origen.get()
        destino = entrada_destino.get()
        if not origen or not destino:
            messagebox.showerror("Error", "Debes seleccionar ambas carpetas")
            return

        carpeta_procesados = os.path.join(origen, "procesados")
        os.makedirs(carpeta_procesados, exist_ok=True)

        organizador = MediaOrganizer(
            src_folder=origen,
            dst_folder=destino,
            processed_folder=carpeta_procesados,
            ffmpeg_path=FFMPEG_EXE,
            ffprobe_path=FFPROBE_EXE,
            processed_log_path=REGISTRO_JSON
        )
        organizador.organize()
        messagebox.showinfo("Éxito", "¡Organización completada!")

    root = tk.Tk()
    root.title("Organizador de Medios")

    entrada_origen = tk.StringVar()
    entrada_destino = tk.StringVar()

    tk.Label(root, text="Carpeta de origen:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    tk.Entry(root, textvariable=entrada_origen, width=50).grid(row=0, column=1)
    tk.Button(root, text="Examinar", command=seleccionar_origen).grid(row=0, column=2, padx=5)

    tk.Label(root, text="Carpeta de destino:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    tk.Entry(root, textvariable=entrada_destino, width=50).grid(row=1, column=1)
    tk.Button(root, text="Examinar", command=seleccionar_destino).grid(row=1, column=2, padx=5)

    tk.Button(root, text="Iniciar Organización", command=ejecutar_organizador, bg="green", fg="white").grid(row=2, column=1, pady=15)

    root.mainloop()


if __name__ == "__main__":
    asegurar_ffmpeg()
    iniciar_gui()
