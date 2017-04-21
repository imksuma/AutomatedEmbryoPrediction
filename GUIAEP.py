import tkinter as tk
from tkinter.filedialog import askdirectory
from LS import *
from skimage.io import imread
from os.path import exists

class Application(tk.Frame):

    def __init__(self, master=None,retrain=False):
        super().__init__(master)
        self.pack()
        self.img = None
        self.create_widgets()
        self.machine = LCRF()

    def create_widgets(self):
        self.search = tk.Button(self, text="browse",
                              command=self.getName)
        self.search.pack(side="right")
        self.image = tk.Label(self, image=self.img)
        self.image["text"] = "choose the sequance"
        # self.image["image"] = self.img
        self.image.pack(side="left")

    def readAndValidateImg(self, fname):
        if fname.find("Frame") != -1 and fname.find(".png") != -1:
            img = imread(fname, as_grey=False)
            if img.shape.__len__() == 2:
                return img
            else:
                return None
        else:
            return None

    def getName(self):
        path = askdirectory()
        if exists(path):
            fullPath = []
            for fName in listdir(path):
                fullPath.extend([path+"\\"+fName])

            n_img = 0
            n_pred = 0

            self.changePhotoAndLable(fileName=fullPath[0],pred=0)
            for fileName in fullPath:
                img = self.readAndValidateImg(fileName)
                if img is not None:
                    self.machine.add_new_img(img)
                    n_img += 1

                    pp = self.machine.pred_img(n_pred)
                    if pp is not None:
                        self.changePhotoAndLable(fileName=fullPath[n_pred],pred=pp)
                        n_pred+=1

            self.machine.set_params(seq_is_complete=True)
            while n_pred != n_img:
                pp = self.machine.pred_img(n_pred)
                if pp is not None:
                    self.changePhotoAndLable(fileName=fullPath[n_pred],pred=pp)
                    n_pred+=1

    def changePhotoAndLable(self,fileName,pred):
        self.img = tk.PhotoImage(file=fileName)
        self.image["image"] = self.img

        prediction = pred
        self.search["text"] = prediction

        self.image.pack()
        self.image.update_idletasks()
        self.image.update()

if __name__ == "__main__":
        root = tk.Tk()
        app = Application(master=root)
        app.mainloop()