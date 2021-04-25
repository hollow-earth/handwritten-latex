import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter import scrolledtext
from PIL import ImageTk,Image 
import execute

def updateText(text):
    text_widget["state"] = tk.NORMAL
    text_widget.insert(tk.INSERT, text)
    text_widget["state"] = tk.DISABLED

def getFile():
    global filename
    filename = askopenfilename()
    if ".png" not in filename.lower() and ".jpg" not in filename.lower():
        updateText("Not a valid image, please use a jpg or png.\n")
    else:
        updateText("Valid image found at {file} \n".format(file=str(filename)))
        img = Image.open(filename)
        newHeight = 300/img.size[0]
        img = img.resize((299, int(newHeight*img.size[1])), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img) 
        untreatedImage.configure(image=img)
        untreatedImage.image = img
        untreatedImage.pack(side=tk.LEFT)

def executeProgram():
    if filename == "":
        updateText("Please select an image first.\n")
    else:
        global processedFlag
        global processedImage
        thresholdValue = thresholdEntry.get()
        if type(int(thresholdValue)) is not int:
            updateText("Value is not an integer.\n")
            raise ValueError("Value is not an integer.")
        elif type(int(thresholdValue)) is int and (int(thresholdValue) > 255 or int(thresholdValue) < 0):
            updateText("Value of threshold should be 0-255.\n")
            raise ValueError("Value of threshold should be 0-255.")
        processedFlag = True
        img = execute.processImage(filename, int(thresholdValue))
        processedImage = img
        img = Image.fromarray(img)
        newHeight = 300/img.size[0]
        img = img.resize((299, int(newHeight*img.size[1])), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img) 
        procImage.configure(image=img)
        procImage.image = img
        procImage.pack(side=tk.RIGHT)
        updateText("Processed image has been generated.\n")

def showImage():
    def destroyWindow():
        nwin.destroy()
        showImageButton["state"], okButton["state"], downloadButton["state"] = tk.NORMAL, tk.NORMAL, tk.NORMAL

    if filename == "":
        updateText("Please select an image first.\n")
    elif processedFlag == False:
        updateText("Please process an image first.\n")
    else:
        global processedImage
        nwin = tk.Toplevel()
        nwin.title("Processed Image")
        photo3 = Image.fromarray(processedImage)
        photo2 = ImageTk.PhotoImage(photo3) 
        nwinCanvas = tk.Canvas(nwin, width = photo3.size[0], height = photo3.size[1])
        nwinCanvas.pack(expand = tk.YES, fill = tk.BOTH)
        showImageButton["state"], okButton["state"], downloadButton["state"] = tk.DISABLED, tk.DISABLED, tk.DISABLED
        nwinCanvas.create_image(1, 1, image = photo2, anchor = tk.NW)
        nwin.resizable(True, True)
        nwin.protocol("WM_DELETE_WINDOW", destroyWindow)
        nwin.mainloop()

filename = ""
processedFlag = False

root = tk.Tk()
root.title("Handwriting to LaTeX")
root.geometry("600x600")
root.resizable(False, False)

canvas = tk.Canvas(root, height=624, width = 600, bg="#e6e6e6")
canvas.pack()

imageFrame = tk.Frame(root)
imageFrame.place(relwidth=1, relheight=0.6, rely=0.1)

frame = tk.Frame(root)
frame.place(relwidth=1, relheight=0.1)
uploadImage = tk.PhotoImage(file="./UI/upload.png")
okImage = tk.PhotoImage(file="./UI/ok.png")
magnifyImage = tk.PhotoImage(file="./UI/magnify.png")
downloadImage = tk.PhotoImage(file="./UI/download.png")

downloadButton = tk.Button(frame, bg = "#ff8080", command=getFile, image=downloadImage, relief="flat", width=150, compound="left")
okButton = tk.Button(frame, bg = "#91FF80", command=executeProgram, image=okImage, relief="flat", width=150, compound="left")
showImageButton = tk.Button(frame, bg = "#73FFFB", command=showImage, image=magnifyImage, relief="flat", width=150, compound="left")
updateButton = tk.Button(frame, bg = "#FFFF99", image=uploadImage, relief="flat", width=150, compound="left")
downloadButton.place(relx = 0, rely = 0)
okButton.place(relx = 0.50, rely = 0)
showImageButton.place(relx = 0.25, rely=0)
updateButton.place(relx=0.75, rely=0)

textFrame = tk.Frame(root)
textFrame.place(relwidth=1, relheight=0.3, rely=0.7)
text_widget = tk.Text(textFrame, width=100, height=9, padx=3, pady=3)
text_widget.pack(side=tk.LEFT)
text_widget.insert(tk.INSERT, "Waiting for image input...\n")
text_widget["state"] = tk.DISABLED

procImage = tk.Label(imageFrame, width=298, height=372, pady = 1, padx=1)
untreatedImage = tk.Label(imageFrame, width=298, height=372, pady = 1, padx=1)

thresholdEntry = tk.Entry(imageFrame, bg="#91FF80")
thresholdEntry.place(relx=0.96, rely=0, width=25, height=20)

root.mainloop()