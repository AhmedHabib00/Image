import itertools
from PyQt5.QtWidgets import QWidget, QApplication, QFileDialog, QVBoxLayout, QMessageBox
import sys
import ui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pydicom
import cv2
from PIL import Image
import numpy as np
from functools import partial
import logging
import concurrent.futures
import threading
import operator
import scipy.fftpack
from matplotlib.widgets import RectangleSelector
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon
import math
from scipy.interpolate import interp1d

logging.basicConfig(filename='Errors.log', level=logging.ERROR, format='%(asctime)s:%(name)s:%(message)s')
mode_to_bpp = {'1': 1, 'L': 8, 'P': 8, 'RGB': 24, 'RGBA': 32, 'CMYK': 32, 'YCbCr': 24, 'I': 32, 'F': 32}


# dictionary of bit depth associated with each color mode of an image to get the bit depth of an image by its mode

class MatplotWidget(QWidget):
    def __init__(self, x=1, y=1, parent=None):
        super(MatplotWidget, self).__init__(parent)
        self.figure = Figure(figsize=(x, y))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axis = self.figure.add_subplot()
        self.layoutvertical = QVBoxLayout(self)
        self.layoutvertical.addWidget(self.canvas)


class MainWidget(QWidget, ui.Ui_Form):
    def __init__(self):
        super(MainWidget, self).__init__()
        self.noises = []
        self.setupUi(self)
        self.init_widget()
        self.pushButton.clicked.connect(self.browse)
        self.pushButton_2.clicked.connect(self.ref_frame)
        self.pushButton_4.clicked.connect(self.whitebg)
        self.lineEdit.textChanged.connect(self.zoom)
        self.pushButton_3.clicked.connect(self.filter)
        self.pushButton_5.clicked.connect(self.fourier_2)
        self.pushButton_7.clicked.connect(self.check_noise)
        self.pushButton_6.clicked.connect(self.check_filter)

    def gen_layout(self, x):
        '''
        A function to add the widget to my GUI to make the initiate function easier and smaller.
        :param x: The widget to be added to the GUI.
        :return:
        '''
        w = MatplotWidget()
        l = QVBoxLayout(x)
        l.addWidget(w)
        return w, l

    def init_widget(self):
        """
        A function to initiate the widgets for display.
        :return:
        """
        self.img_widget, self.layoutverical = self.gen_layout(self.widget)
        self.widget2, self.layoutvertical2 = self.gen_layout(self.widget_2)
        self.widget3, self.layoutvertical3 = self.gen_layout(self.widget_3)
        self.widget4, self.layoutvertical4 = self.gen_layout(self.widget_7)
        self.uneqhist, self.layoutvertical5 = self.gen_layout(self.widget_6)
        self.eqhist, self.layoutvertical6 = self.gen_layout(self.widget_8)
        self.uneqimg, self.layoutvertical7 = self.gen_layout(self.widget_4)
        self.eqimg, self.layoutvertical8 = self.gen_layout(self.widget_5)
        self.org_img, self.layoutvertical9 = self.gen_layout(self.widget_9)
        self.filtered_img, self.layoutvertical10 = self.gen_layout(self.widget_10)
        self.fft_org_img, self.layoutvertical11 = self.gen_layout(self.widget_11)
        self.fft_mag, self.layoutvertical12 = self.gen_layout(self.widget_12)
        self.fft_phase, self.layoutvertical13 = self.gen_layout(self.widget_13)
        self.ifft_img, self.layoutvertical14 = self.gen_layout(self.widget_14)
        self.diff_ifft_img, self.layoutvertical15 = self.gen_layout(self.widget_15)
        self.npimg, self.layoutvertical16 = self.gen_layout(self.widget_16)
        self.roi, self.layoutvertical17 = self.gen_layout(self.widget_17)
        self.roi_hist, self.layoutvertical18 = self.gen_layout(self.widget_18)
        self.filtered_roi, self.layoutvertical25 = self.gen_layout(self.widget_25)
        self.shepp_logan, self.layoutvertical19 = self.gen_layout(self.widget_19)
        self.sinogram, self.layoutvertical20 = self.gen_layout(self.widget_20)
        self.laminogram_1, self.layoutvertical21 = self.gen_layout(self.widget_21)
        self.laminogram_2, self.layoutvertical22 = self.gen_layout(self.widget_22)
        self.laminogram_ram_lak, self.layoutvertical23 = self.gen_layout(self.widget_23)
        self.laminogram_hamming, self.layoutvertical24 = self.gen_layout(self.widget_24)
        self.ca_img, self.layoutvertical25 = self.gen_layout(self.widget_26)
        self.ca_roi, self.layoutvertical26 = self.gen_layout(self.widget_27)
        self.ca_cm, self.layoutvertical27 = self.gen_layout(self.widget_28)

    def zoom(self):
        """
        A function to perform the zoom, using both bilinear and nearest neighbor interpolation and display them in the second tab
        in both widgets I allocated for the zoomed images.
        :return:
        """
        try:
            if self.lineEdit.text() != "":
                with concurrent.futures.ThreadPoolExecutor() as executer:
                    executer.submit(partial(self.bilinear_interpolation, float(self.lineEdit.text())))
                    executer.submit(partial(self.nearest_neighbor_interpolation, float(self.lineEdit.text())))

                # t1 = threading.Thread(target=partial(self.bilinear,int(self.lineEdit.text())))
                # t2 = threading.Thread(target=partial(self.nn_inter,int(self.lineEdit.text())))
                # t1.start()
                # t2.start()
                # t2.join()
        except Exception as e:
            logging.error(f'An error occured in the zoom function (line 60) and it is : {e}')
            print(e)

    def rgb2gray(self, x):
        try:
            if x == 1:
                # self.gimg =self.gray_img = np.dot(self.im[..., :3], [0.299, 0.587, 0.114])
                self.gimg = np.average(self.im, axis=-1)
                self.img_widget.axis.imshow(self.gimg, cmap='gray')
                self.img_widget.canvas.draw()
                self.show_info(0)
            else:
                self.img_widget.axis.imshow(self.img_data)
                self.img_widget.canvas.draw()
        except Exception as e:
            print(e)

    def browse(self):
        '''
        browse function to browse an image and then display it, checking for errors with exception handeling.
        '''
        self.fname = QFileDialog.getOpenFileName(self, "Open Image", ".",
                                                 "JPG images(*.jpg);;Dicom Images (*.dcm);;BMP image(*.bmp);;PNG image(*.png)")
        if self.fname[0] == "":
            return
        try:
            if self.fname[0][-3:] == 'dcm' or self.fname[0][-3:] == 'DCM':
                # if the image is dicom then it should be read with pydicom not like other images.
                self.dic = pydicom.dcmread(self.fname[0])
                arr = self.dic.pixel_array
                self.im = self.img_data = arr

                self.img_widget.axis.imshow(arr, cmap='gray')
                self.img_widget.canvas.draw()
                self.show_info(x=1)
            else:
                self.img_data = Image.open(self.fname[0])
                self.im = cv2.imread(self.fname[0])
                self.img_widget.axis.imshow(self.img_data, cmap='gray')
                self.img_widget.canvas.draw()
                self.roi_flag = False
                self.gray = np.average(self.im, axis=-1)
                self.gray_img = np.dot(self.im[..., :3], [0.299, 0.587, 0.114])
                self.show_info(0)
                self.noise()
                self.color_map()
                t5 = threading.Thread(target=self.back_projection)
                t5.start()
                t4 = threading.Thread(target=self.fourier)
                t4.start()
                t3 = threading.Thread(target=self.histogram_equalization)
                t3.start()
        except Exception as e:
            logging.error(f'An error occured in the browse function (line 98) and it is : {e}')
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('Error! Image Corrupted')
            msg.setWindowTitle('ERROR')
            msg.exec_()
            print(e)
            self.browse()

    def whitebg(self):
        '''
        A function to display a white background image for task 1.5
        '''
        self.ref_im = cv2.imread("whitebg.jpg")
        self.display(self.ref_im, self.widget4)

    def display(self, x, w, y=None):
        if y:
            w.axis.imshow(x, cmap=y)
        w.axis.imshow(x)
        w.canvas.draw()

    def ref_frame(self):
        '''
        Task 1.5 : Color last 4 pixels in 2nd row red and last 4 pixels in 2nd col in blue.
        :return:
        '''
        self.ref_im = cv2.imread("whitebg.jpg")
        self.display(self.ref_im, self.widget4)
        self.ref_im[1, -4:] = [255, 0, 0]
        self.ref_im[-4:, 1] = [0, 0, 255]
        self.widget4.axis.imshow(self.ref_im)
        self.widget4.canvas.draw()

    def show_info(self, x):
        '''
        Function to print the information required for the image.
        :param x: to specify if it's a dicom then x =1 to display the additional info for dicom images.
        '''
        if x == 1:
            # checking each required information if it's included in the dicom or not to avoid crashes and errors
            # if it's not in the dicom data it will print NA.
            self.label.setText(
                f"Image Size: {self.dic.Rows if 'Rows' in self.dic else 0} x {self.dic.Columns if 'Columns' in self.dic else 0} \n"
                f"Image Total Size: {self.dic.Rows * self.dic.Columns * self.dic.BitsAllocated} bits \n"
                f"Bit Depth: {self.dic.BitsAllocated if 'BitsAllocated' in self.dic else 'NA'} \n"
                f"Image Color: {self.dic.PhotometricInterpretation if 'PhotometricInterpretation' in self.dic else 'NA'} \n"
                f"Modality: {self.dic.Modality if 'Modality' in self.dic else 'NA'} \n"
                f"Patient Name: {self.dic.PatientName}"
                f"Patient Age : {self.dic.PatientAge if 'PatientAge' in self.dic else 'NA'} \n"
                f"Body Part Examined: {self.dic.BodyPartExamined if 'BodyPartExamined' in self.dic else 'NA'} \n"
                )
        else:
            self.label.setText(f' Image Size : {self.img_data.size[0]} x {self.img_data.size[1]} \n '
                               f'Image Total Size: {self.img_data.size[0] * self.img_data.size[1] * mode_to_bpp[self.img_data.mode]}  \n'
                               f'Bit Depth : {mode_to_bpp[self.img_data.mode]} \n'
                               f'Image Color : {self.img_data.mode} \n')

    def nearest_neighbor_interpolation(self, x):
        '''
        Function to perform nearest neighbor interpolation of the image
        :param x: zoom factor
        :return:
        '''
        try:
            dpi = self.img_data.info['dpi']
            self.widget3.axis.clear()
            old_width = self.gray.shape[0]
            old_height = self.gray.shape[1]
            new_width = x * old_width
            new_height = x * old_height
            x_scale = new_width / (old_width - 1)
            y_scale = new_height / (old_height - 1)
            new_image = np.zeros((int(new_width), int(new_height)), dtype=np.float32)
            for new_x in range(0, int(new_width)):
                for new_y in range(0, int(new_height)):
                    new_image[new_x, new_y] = self.gray_img[round(new_x / x_scale), round(new_y / y_scale)]
            xi = new_image.shape[0] / dpi[0]
            yi = new_image.shape[1] / dpi[1]
            self.widget3.figure.set_size_inches((xi, yi), forward=True)
            self.widget3.axis.imshow(new_image, cmap='gray')
            # self.widget3.axis.set_xlim(xmin=0, xmax=self.img_data.size[0])
            # self.widget3.axis.set_ylim(ymin=self.img_data.size[1], ymax=0)
            self.widget3.canvas.draw()
        except Exception as e:
            logging.error(f'An error in nearest neighbor interpolation func and it is : {e}')
            print(e)

    def bilinear_interpolation(self, r,rotate=False, img=None, theta=None):
        try:
            if rotate:
                h, w = img.shape
                center = (h // 2, w // 2)
                a = h / 2
                b = w / 2
                new_img = np.zeros((h, w))
                for i in range(0, h):
                    for j in range(0, w):
                        x = (i - center[0]) * np.cos(theta) + (j - center[1]) * np.sin(theta) + a
                        y = -(i - center[0]) * np.sin(theta) + (j - center[1]) * np.cos(theta) + b
                        if x < img.shape[0] and y < img.shape[1] and x > 0 and y > 0:
                            new_img[i, j] = img[int(x), int(y)]
                return new_img

            self.widget2.axis.clear()
            w = self.gray.shape[0]
            h = self.gray.shape[1]
            new_image = np.zeros((int(h * r), int(w * r)), dtype=np.float32)
            for i in range(0, int(h * r)):
                x1 = int(np.floor(i / r))
                x2 = int(np.ceil(i / r))
                if x1 == 0:
                    x1 = 1
                if x2 >= self.gray.shape[0]:
                    x2 = self.gray.shape[0] - 1
                x = np.remainder(i / r, 1)
                for j in range(0, int(w * r)):
                    y1 = int(np.floor(j / r))
                    y2 = int(np.ceil(j / r))
                    if y1 == 0:
                        y1 = 1
                    if y2 >= self.gray.shape[1]:
                        y2 = self.gray.shape[1] - 1
                    ctl = self.gray_img[x1, y1]
                    cbl = self.gray_img[x2, y1]
                    ctr = self.gray_img[x1, y2]
                    cbr = self.gray_img[x2, y2]
                    y = np.remainder(j / r, 1)
                    tr = (ctr * y) + (ctl * (1 - y))
                    br = (cbr * y) + (cbl * (1 - y))
                    new_image[i, j] = (br * x) + (tr * (1 - x))
            dpi = self.img_data.info['dpi']
            xi = new_image.shape[0] / dpi[0]
            yi = new_image.shape[1] / dpi[1]
            self.widget2.figure.set_size_inches((xi, yi), forward=True)
            self.widget2.axis.imshow(new_image, cmap='gray')
            self.widget2.canvas.draw()
        except Exception as e:
            logging.error(f"An error in bilinear interpolation function which is : {e}")
            print(e)

    def histogram_equalization(self, roi=None):
        try:
            if roi is None:
                N = self.im.shape[0] * self.im.shape[1]
            else:
                N = roi.shape[0] * roi.shape[1]
            img_data = Image.open(self.fname[0])
            if img_data.mode == 'RGB' or img_data.mode == 'RGBA':
                im = np.average(self.im, axis=-1)
                img_data = im

            ## GUI
            self.eqimg.axis.clear()
            self.uneqimg.axis.clear()
            self.eqhist.axis.clear()
            self.uneqhist.axis.clear()
            ## ~ GUI

            if roi is None:
                hist_img = np.asarray(img_data)
            else:
                hist_img = np.asarray(roi)
            flat = hist_img.flatten()  # Put Pixels in 1D array
            flat = flat.astype(int)  # make intensities as int to avoid problems of float indexing errors
            nk = np.zeros(256)
            for pixel in flat:
                nk[pixel] += 1  # get count of pixel values
            pk = nk / N
            rk = np.arange(0, 256)
            mean = 0
            for i in range(len(rk)):
                mean += rk[i] * pk[i]
            variance = 0
            for i in range(len(rk)):
                variance += ((rk[i] - mean)**2) * pk[i]
            # GUI
            if roi is not None:
                self.roi_hist.axis.clear()
                self.roi_hist.axis.bar(rk, pk)
                self.roi_hist.canvas.draw()
                self.label_20.setText(f'ROI Mean : {np.round(mean, 3)}')
                self.label_21.setText(f'ROI Standard Deviation : {np.round(np.sqrt(variance), 3)}')
                return
            self.uneqimg.axis.imshow(self.img_data, cmap='gray')
            self.uneqimg.canvas.draw()
            self.uneqhist.axis.bar(rk, pk)
            self.uneqhist.canvas.draw()

            #~ GUI
            v = 0
            CDF = [v := v + n for n in pk]  # assignment operator to simultaneously update my v value
            sk = []

            CDF = np.array(CDF)
            for i in CDF:
                sk.append(np.round(255 * i))
            sk = np.array(sk, dtype=np.uint8)
            equalized_image = sk[flat]  # accessing all the indices of the original image in the new sk array
            equalized_image = np.reshape(equalized_image, (self.im.shape[0], self.im.shape[1]))
            self.eqimg.axis.imshow(equalized_image, cmap='gray')
            self.eqimg.canvas.draw()

            # creating a list to zip the sk and the pk together
            temp = list(zip(sk, pk))
            ps = []
            sk = []
            # group by my first index of the tuple temp which is my sk to get the sum of the pk corresponding to the indices
            for key, group in itertools.groupby(temp, operator.itemgetter(0)):
                # print(key, [t[-1] for t in group])
                sk.append(key)
                ps.append(sum(t[-1] for t in group))

            # thought of padding the sk and ps arrays with zeros to match the 256 length of my
            # histogram but the bar plot will fill the gaps with zeros in the plot anyway.

            # num = [c for c,e in enumerate(self.sk) if e>=self.ps[0]][0]
            # c= [0 for e in range(num)]+self.ps+[0 for e in range(len(self.sk)-num-len(self.ps))]
            ps = np.array(ps)
            self.eqhist.axis.bar(sk, ps)
            self.eqhist.canvas.draw()
        except Exception as e:
            logging.error(f'An error occured in the zoom function (line 281) and it is : {e}')
            print(e)

    def pad(self, im, kernel_width, kernel_height, combobox=None):
        img_width = im.shape[0]
        img_height = im.shape[1]
        padded_img = np.zeros((img_width + (kernel_width - 1), img_height + (kernel_height - 1)))
        if combobox is None:
            padded_img[(kernel_width // 2):-(kernel_width // 2), (kernel_height // 2):-(kernel_height // 2)] = im
        else:
            if combobox.currentText() == 'Zero':
                padded_img[(kernel_width // 2):-(kernel_width // 2), (kernel_height // 2):-(kernel_height // 2)] = im

            elif combobox.currentText() == 'Mirror':
                padded_img = np.concatenate((im[:, (kernel_width - 1):0:-1], im), axis=1)
                padded_img = np.concatenate(
                    (padded_img, im[:, len(padded_img):len(padded_img) - (kernel_width - 1):-1]), axis=1)
                padded_img = np.concatenate(
                    (padded_img, padded_img[padded_img.shape[0] - 2:-padded_img.shape[0] - (kernel_width - 1):-1, :]),
                    axis=0)
                padded_img = np.concatenate((padded_img[kernel_width - 1:0:-1, :], padded_img), axis=0)

            elif combobox.currentText() == 'Replicate':
                padded_img[0:(kernel_width // 2) + 1, :] = padded_img[kernel_width - 1, :]
                padded_img[-(kernel_width - 1):, :] = padded_img[len(padded_img) - kernel_width, :]
                for i in range(kernel_width - 1, -1, -1):
                    padded_img[:, i] = padded_img[:, kernel_width - 1]
                for i in range(padded_img.shape[1] - kernel_width, padded_img.shape[1]):
                    padded_img[:, i] = padded_img[:, len(padded_img) - kernel_width]
        return padded_img

    def convolve(self, kernel, kernel_size, img):
        out_img = np.zeros_like(img)
        img_width, img_height = img.shape
        kernel_height = kernel_width = kernel_size
        img = self.pad(img, kernel_width, kernel_height)
        for x in range(img_height):
            for y in range(img_width):
                out_img[y, x] = np.sum((kernel * img[y:y + kernel_width, x:x + kernel_height]))
        return out_img

    def filter(self, kernel_size=None, ret=False, combo=0):
        try:

            img = self.gray
            #self.img_data = self.gray
            img_width = img.shape[0]
            img_height = img.shape[1]
            if kernel_size:
                kernel_width = kernel_size
                kernel_height = kernel_size
            else:
                kernel_width = int(self.lineEdit_2.text())
                kernel_height = int(self.lineEdit_4.text())
            if kernel_height % 2 == 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText('Error! Use odd values for the box filter')
                msg.setWindowTitle('ERROR')
                msg.exec_()
                raise Exception('Use odd kernel size')
            img = np.array(self.gray)
            kernel = np.ones((kernel_width, kernel_height), dtype=np.float32)/(kernel_height*kernel_width)
            if combo == 0:
                padded_img = self.pad(img, kernel_width, kernel_height, self.comboBox)
            else:
                padded_img = self.pad(img, kernel_width, kernel_height, self.comboBox_2)
            self.org_img.axis.clear()
            self.filtered_img.axis.clear()
            im = img.astype('float')
            out_img = np.zeros_like(im)
            for x in range(img_height):
                for y in range(img_width):
                    out_img[y, x] = np.sum((kernel*padded_img[y:y+kernel_width, x:x+kernel_height]))
            ret_img = out_img
            diff_img = img - out_img
            self.org_img.axis.imshow(abs(diff_img), cmap='gray')
            self.org_img.canvas.draw()
            if not self.lineEdit_3:
                pass
            else:
                if not kernel_size:
                    out_img = float(self.lineEdit_3.text()) * diff_img
                else:
                    out_img = float(kernel_size) * diff_img
                #self.out_img = np.array(self.out_img)
                #self.out_img = self.out_img.astype('float')
                out_img += img
                out_img = self.scale(out_img)
                self.filtered_img.axis.imshow(out_img,cmap='gray')
                self.filtered_img.canvas.draw()
            if ret:
                out_img = self.scale(out_img)
                return kernel, ret_img
        except Exception as e:
            print(e)

    def fourier(self, img=None, ret=False):
        try:
            if img is None:
                img_fft = scipy.fftpack.fftshift(scipy.fftpack.fft2(self.gray))
            else:
                img_fft = scipy.fftpack.fftshift(scipy.fftpack.fft2(img))
            mag = np.log(np.abs(img_fft))
            phase = np.angle(img_fft)
            self.fft_org_img.axis.imshow(self.img_data, cmap='gray')
            self.fft_org_img.canvas.draw()
            self.fft_mag.axis.imshow(mag, cmap='gray')
            self.fft_mag.canvas.draw()
            self.fft_phase.axis.imshow(phase, cmap='gray')
            self.fft_phase.canvas.draw()
            if ret:
                return img_fft
        except Exception as e:
            print(e)

    def fourier_2(self):
        try:
            flag = False
            if self.im.shape[0] % 2 !=0:
                temp = np.zeros((self.im.shape[0]+1,self.im.shape[1]))
                temp[0:self.im.shape[0], 0:self.im.shape[1]] = self.im
                gray = temp
                self.im = temp
                self.img_data = temp
                flag = True
            if self.im.shape[1] % 2 != 0:
                temp = np.zeros((self.im.shape[0],self.im.shape[1]+1))
                temp[0:self.im.shape[0], 0:self.im.shape[1]] = self.im
                gray = temp
                self.im = temp
                self.img_data = temp
                flag = True
            if flag:
                img_fourier = self.fourier(img=gray, ret=True)
            else:
                img_fourier = self.fourier(ret=True)

            org_kernel, spat_filt_img = self.filter(kernel_size=int(self.lineEdit_5.text()), ret=True, combo=1)
            kernel_size = int(self.lineEdit_5.text())
            kernel = np.zeros((self.im.shape[0], self.im.shape[1]))
            kernel[((img_fourier.shape[0] - kernel_size + 1) // 2):-((img_fourier.shape[0]-kernel_size) // 2), ((img_fourier.shape[1] - kernel_size +1) // 2):-((img_fourier.shape[1]-kernel_size) // 2)] = org_kernel
            kernel = scipy.fftpack.ifftshift(kernel)
            kernel_fourier = self.fourier(img=kernel, ret=True)
            freq_filtered = kernel_fourier * img_fourier
            freq_filtered = np.real(scipy.fftpack.ifft2(scipy.fftpack.fftshift(freq_filtered)))
            self.ifft_img.axis.imshow(freq_filtered, cmap='gray')
            self.ifft_img.canvas.draw()
            diff_img = freq_filtered - spat_filt_img

            self.diff_ifft_img.axis.imshow(diff_img, cmap='gray')
            self.diff_ifft_img.canvas.draw()

            # the difference was 0 - a black image - which means that we got the same result but it was much faster,
            # simpler to code using the fourier transform and filtering in the frequency domain than it was in the
            # spatial domain.
        except Exception as e:
            logging.error(f'An error occured in the fourier_2 function (line 530) and it is : {e}')
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(f'{e}')
            msg.setWindowTitle('ERROR')
            msg.exec_()

    def scale(self, g):
        gm = g - g.min()
        gs = 255 * (gm / gm.max())
        return gs

    def check_noise(self):
        check_boxes = [self.checkBox, self.checkBox_2, self.checkBox_3, self.checkBox_4, self.checkBox_5, self.checkBox_6]
        self.noises = []
        for box in check_boxes:
            if box.isChecked():
                self.noises.append(box.text())
        self.noise()

    def contraharmonic_mean(self, q, img):
        """"
        if q is 0 it reduces to arithmetic mean filter
        and if q = -1 it reduces to harmonic filter
        :param q:
        :param img:
        :return:
        """
        num = np.power(img, q+1)
        denom = np.power(img, q)
        kernel = np.full((3, 3), 1.0)
        filtered_img = self.convolve(kernel, kernel.shape[0], num) / self.convolve(kernel, kernel.shape[0], denom)
        return filtered_img

    def geometric_mean(self, img):
        kernel_size = 5
        img = self.pad(img, kernel_size, kernel_size)
        filtered_img = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                filtered_img[i, j] = np.prod(img[i:i+kernel_size, j:j+kernel_size])**(1/(kernel_size**2))
        return filtered_img

    def calc_mean_var(self, roi):
        try:
            N = roi.shape[0] * roi.shape[1]
            hist_img = np.asarray(roi)
            flat = hist_img.flatten()
            flat = flat.astype(int)
            nk = np.zeros(256)
            for pixel in flat:
                nk[pixel] += 1  # get count of pixel values
            pk = nk / N
            rk = np.arange(0, 256)
            mean = 0
            for i in range(len(rk)):
                mean += rk[i] * pk[i]
            variance = 0
            for i in range(len(rk)):
                variance += ((rk[i] - mean)**2) * pk[i]
            return mean, variance
        except Exception as e:
            print(e)

    def adaptive_filter(self, img):
        img_mean, img_variance = self.calc_mean_var(img)
        img = self.pad(img, 7, 7)
        kernel = np.ones((7, 7)) / 49
        filt = np.zeros_like(img)
        c = 0
        z = 0
        for i in range(7, img.shape[0]):
            for j in range(7, img.shape[1]):
                window = img[i-3:i+3, j-3:j+3]
                roi_filter = window.flatten()
                local_mean, local_variance = self.calc_mean_var(window)
                local_mean = np.mean(window)
                if img_variance == 0:
                    val = img[i, j]
                elif local_variance > img_variance:
                    c += 1
                    val = img[i, j] - (img_variance / local_variance) * (img[i, j] - local_mean)
                elif local_variance == img_variance:
                    val = local_mean
                elif img_variance > local_variance:
                    z += 1
                    val = img[i, j] - (img[i, j] - local_mean)
                filt[i, j] = val
        print(c, z)
        return filt

    def median_filter(self, img):
        temp = []
        kernel_size = 7
        index = kernel_size // 2
        output_img = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for z in range(kernel_size):
                    if i + z - index < 0 or i + z - index > img.shape[0] - 1:
                        for _ in range(kernel_size):
                            temp.append(0)
                    else:
                        if j +z - index < 0 or j + index > img.shape[1] - 1:
                            temp.append(0)
                        else:
                            for k in range(kernel_size):
                                temp.append(img[i + z - index, j + k - index])
                temp.sort()
                output_img[i, j] = temp[len(temp) // 2]
                temp = []
        return output_img

    def check_filter(self):
        try:
            check_boxes = [self.checkBox_10, self.checkBox_11, self.checkBox_12, self.checkBox_7, self.checkBox_8, self.checkBox_9]
            filters = []
            for box in check_boxes:
                if box.isChecked():
                    filters.append(box.text())
            for i in filters:
                if i == 'Mean':
                    filtered_img = self.contraharmonic_mean(q=0, img=self.noisy_roi)
                elif i == 'Harmonic':
                    filtered_img = self.contraharmonic_mean(q=-1, img=self.noisy_roi)
                elif i == 'Contraharmonic':
                    filtered_img = self.contraharmonic_mean(q=1.5, img=self.noisy_roi)
                    filtered_img = self.contraharmonic_mean(q=-1.5, img=filtered_img)
                elif i == 'Geometric':
                    filtered_img = self.geometric_mean(img=self.noisy_roi)
                elif i == 'Adaptive':
                    filtered_img = self.adaptive_filter(self.noisy_roi)
                elif i == 'Median':
                    filtered_img = self.median_filter(self.noisy_roi)

            self.filtered_roi.axis.imshow(filtered_img, cmap='gray')
            self.filtered_roi.canvas.draw()
        except Exception as e:
            print(e)

    def gen_noise(self, noises, img):
        try:
            noise = 0
            h, w = img.shape
            if not len(noises):
                return np.zeros((h,w))
            for kind in noises:
                if kind == 'gaussian':
                    noise += np.random.normal(loc=0.0, scale=1000, size=(h, w))
                if kind == 'uniform':
                    noise += np.random.uniform(-10, 10, (h, w))
                if kind == 'rayleigh':
                    noise += np.random.rayleigh(size=img.shape)
                if kind == 'exponential':
                    noise += np.random.exponential(10, size=img.shape)
                if kind == 'gamma':
                    noise += np.random.gamma(5, size=img.shape)
                if kind == 'salt_n_pepper':
                    salt_percent = 0.1
                    pepper_percent = 0.1
                    percentile = np.random.rand(img.shape[0], img.shape[1])
                    pepper = np.where(percentile < pepper_percent)
                    salt = np.where(percentile < salt_percent)
                    noise = img
                    noise[salt] = 255
                    noise[pepper] = 0
            noise = self.scale(noise)
            return noise
        except Exception as e:
            print(e)

    def select_roi(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        points = [x1, x2, y1, y2]
        for i in points:
            if i < 0:
                i = 0
            if i > 255:
                i = 255
        roi = self.noisy_img[int(min(points[2:])):int(max(points[2:])), int(min(points[:2])):int(max(points[:2]))]
        self.noisy_roi = roi
        self.roi.axis.imshow(roi, cmap='gray')
        self.roi.canvas.draw()
        self.histogram_equalization(roi)

    def toggle_selector(self, event):
        print('key pressed. ')
        if event.key in ['Q', 'q'] and self.toggle_selector_RS.active:
            print('deactivated')
            self.toggle_selector_RS.set_active(False)
        if event.key in ['R', 'r'] and not self.toggle_selector_RS.active:
            print('activated')
            self.toggle_selector_RS.set_active(True)
        print(event.key)

    def noise(self):
        img = np.ones((256, 256), dtype=np.float32) * 50  # black square
        x_edge_gray = int(len(img) / 6)
        img[x_edge_gray:-x_edge_gray, x_edge_gray:-x_edge_gray] = 150  # gray square
        center = (int(len(img)/2), int(len(img)/2))
        radius = min(x_edge_gray, x_edge_gray, len(img)-center[0], len(img)-center[1])
        x, y = np.ogrid[:len(img), :len(img)]
        dist = np.sqrt((x - center[0])**2 + (y - center[0])**2)
        circle = dist <= radius
        img[circle] = 250
        noise = self.gen_noise(self.noises, img)
        print('aloo')
        noisy_img = np.add(img, noise)
        self.noisy_img = self.scale(noisy_img)
        self.toggle_selector_RS_2 = RectangleSelector(self.npimg.axis, self.select_roi, drawtype='box', useblit=True,\
                                               button=[1, 3], spancoords='pixels', interactive=True)
        self.npimg.axis.imshow(noisy_img, cmap='gray')
        self.npimg.canvas.mpl_connect('key_press_event', self.toggle_selector)
        self.npimg.canvas.draw()

    def inverse_radon(self, img, theta):
        try:
            n = len(theta)
            img = img.astype(np.float32)
            output_size = int(img.shape[0])
            reconstructed = np.zeros((output_size, output_size), dtype=np.float32)
            radius = output_size // 2
            xpr, ypr = np.mgrid[:output_size, :output_size] - radius
            x = np.arange(img.shape[0]) - img.shape[0] // 2
            for col, angle in zip(np.transpose(img), theta):
                angle = math.radians(angle)
                t = ypr * np.cos(angle) - xpr * np.sin(angle)
                rotat = interp1d(x, col, bounds_error=False, fill_value=0)
                reconstructed += rotat(t)
            return reconstructed * np.pi / (2 * n)
        except Exception as e:
            print(e)

    def radon_transform(self, img, theta):
        try:
            img = img.astype(np.float32)
            radon_img = np.zeros((img.shape[0], len(theta)), dtype=np.float32)
            for i, t in enumerate(theta):
                t = -1 * math.radians(t)
                rotated = self.bilinear_interpolation(1, rotate=True, img=img, theta=t)
                radon_img[:, i] = rotated.sum(0)
            return radon_img
        except Exception as e:
            print(e)

    def back_projection(self):
        img = shepp_logan_phantom()
        img = rescale(img, scale=0.64, mode='reflect')
        self.shepp_logan.axis.imshow(img, cmap='gray')
        self.shepp_logan.canvas.draw()
        theta = np.linspace(0, 180, 180)
        sinogram = radon(img, theta=theta)
        self.sinogram.axis.imshow(sinogram, cmap='gray')
        self.sinogram.canvas.draw()
        sinogram_1 = radon(img, theta=[0, 20, 40, 60, 160])
        #sino = self.radon_transform(img, theta=theta)
        lam = self.inverse_radon(sinogram, theta=theta)
        self.laminogram_1.axis.imshow(lam, cmap='gray')
        self.laminogram_1.canvas.draw()
        self.laminogram_2.axis.imshow(iradon(sinogram, filter=None), cmap='gray')
        self.laminogram_2.canvas.draw()
        self.laminogram_ram_lak.axis.imshow(iradon(sinogram, theta=theta, filter='ramp'), cmap='gray')
        self.laminogram_ram_lak.canvas.draw()
        self.laminogram_hamming.axis.imshow(iradon(sinogram, theta=theta, filter='hamming'), cmap='gray')
        self.laminogram_hamming.canvas.draw()

    def select_roi_cmap(self, eclick, erelease):
        try:
            self.rgb_img_cm = cv2.imread('US_Image.jpeg')
            #self.rgb_img_cm = cv2.cvtColor(self.rgb_img_cm, cv2.COLOR_BGR2RGB)
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            points = [x1, x2, y1, y2]
            for i in points:
                if i < 0:
                    i = 0
                if i > 255:
                    i = 255
            roi_cmap = self.rgb_img_cm[int(min(points[2:])):int(max(points[2:])), int(min(points[:2])):int(max(points[:2]))]
            self.ca_cm.axis.cla()
            self.ca_roi.axis.cla()
            img_cm = roi_cmap
            img_cm = np.average(img_cm, axis=-1)
            padded_cm = np.pad(img_cm, 25, mode='edge')
            avg_mat = np.zeros_like(img_cm)
            for x in range(img_cm.shape[0]):
                for y in range(img_cm.shape[1]):
                    avg_mat[x, y] = np.sum((padded_cm[x:x + 25, y:y + 25])) / 625
            sq_cm_img = img_cm ** 2
            sq_pad = np.pad(sq_cm_img, 25, mode='edge')
            sq_avg = np.zeros_like(sq_cm_img)
            for x in range(sq_cm_img.shape[0]):
                for y in range(sq_cm_img.shape[1]):
                    sq_avg[x, y] = np.sum((sq_pad[x:x + 25, y:y + 25])) / 625
            local_variance = sq_avg - (avg_mat ** 2)
            local_variance = self.scale(local_variance)
            cmapped = cv2.applyColorMap(np.uint8(local_variance), cv2.COLORMAP_JET)
            temp = self.rgb_img_cm
            img = temp
            cmapped = cv2.cvtColor(cmapped, cv2.COLOR_BGR2RGB)

            img[int(min(points[2:])):int(max(points[2:])), int(min(points[:2])):int(max(points[:2]))] = cmapped
            self.ca_roi.axis.imshow(img)
            self.ca_roi.canvas.draw()
            self.ca_cm.axis.imshow(cmapped)
            self.ca_cm.canvas.draw()

            # self.ca_cm.axis.imshow(cmapped, cmap='jet')
            # pc = self.ca_cm.axis.pcolor(local_variance)
            # self.ca_cm.figure.colorbar(pc)
        except Exception as e:
            print(e)

    def color_map(self):
        try:
            self.rgb_img_cm = cv2.imread('US_Image.jpeg')
            #self.rgb_img_cm = np.stack((img,)*3, axis=-1)
            self.ca_img.axis.imshow(self.rgb_img_cm, cmap='gray')
            self.ca_img.canvas.draw()
            self.toggle_selector_RS = RectangleSelector(self.ca_img.axis, self.select_roi_cmap, drawtype='box', useblit=False, \
                                                        button=[1, 3], spancoords='pixels', interactive=True)
            self.ca_img.canvas.mpl_connect('key_press_event', self.toggle_selector)
        except Exception as e:
            logging.error(e)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWidget()
    w.show()
    sys.exit(app.exec_())