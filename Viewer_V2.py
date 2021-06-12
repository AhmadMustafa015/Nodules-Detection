import math
import warnings
import tkinter as tk
from tkinter import *
import os
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
from tkinter import filedialog,messagebox
import SimpleITK
import helpers
import numpy
from bs4 import BeautifulSoup
import pandas
import glob
import shutil
import copy

class MousePositionTracker(tk.Frame):
    """ Tkinter Canvas mouse position widget. """

    def __init__(self, canvas):
        self.canvas = canvas
        self.canv_width = self.canvas.cget('width')
        self.canv_height = self.canvas.cget('height')
        self.reset()

        # Create canvas cross-hair lines.
        xhair_opts = dict(dash=(3, 2), fill='white', state=tk.HIDDEN)
        self.lines = (self.canvas.create_line(0, 0, 0, self.canv_height, **xhair_opts),
                      self.canvas.create_line(0, 0, self.canv_width,  0, **xhair_opts))

    def cur_selection(self):
        return (self.start, self.end)

    def begin(self, event):
        self.hide()
        self.start = (event.x, event.y)  # Remember position (no drawing).

    def update(self, event):
        self.end = (event.x, event.y)
        self._update(event)
        self._command(self.start, (event.x, event.y))  # User callback.

    def _update(self, event):
        # Update cross-hair lines.
        self.canvas.coords(self.lines[0], event.x, 0, event.x, self.canv_height)
        self.canvas.coords(self.lines[1], 0, event.y, self.canv_width, event.y)
        self.show()

    def reset(self):
        self.start = self.end = None

    def hide(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.HIDDEN)
        self.canvas.itemconfigure(self.lines[1], state=tk.HIDDEN)

    def show(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.NORMAL)
        self.canvas.itemconfigure(self.lines[1], state=tk.NORMAL)

    def autodraw(self, command=lambda *args: None):
        """Setup automatic drawing; supports command option"""
        self.reset()
        self._command = command
        self.canvas.bind("<Button-1>", self.begin)
        self.canvas.bind("<B1-Motion>", self.update)
        self.canvas.bind("<ButtonRelease-1>", self.quit)

    def quit(self, event):
        self.hide()  # Hide cross-hairs.
        self.reset()


class SelectionObject:
    """ Widget to display a rectangular area on given canvas defined by two points
        representing its diagonal.
    """
    def __init__(self, canvas, select_opts):
        # Create attributes needed to display selection.
        self.canvas = canvas
        self.select_opts1 = select_opts
        self.width = self.canvas.cget('width')
        self.height = self.canvas.cget('height')
        self.bbox = []
        # Options for areas outside rectanglar selection.
        select_opts1 = self.select_opts1.copy()  # Avoid modifying passed argument.
        select_opts1.update(state=tk.HIDDEN)  # Hide initially.
        # Separate options for area inside rectanglar selection.
        select_opts2 = dict(dash=(2, 2), fill='', outline='white', state=tk.HIDDEN)

        # Initial extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = 0, 0,  1, 1
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height

        self.rects = (
            # Area *outside* selection (inner) rectangle.
            self.canvas.create_rectangle(omin_x, omin_y,  omax_x, imin_y, **select_opts1),
            self.canvas.create_rectangle(omin_x, imin_y,  imin_x, imax_y, **select_opts1),
            self.canvas.create_rectangle(imax_x, imin_y,  omax_x, imax_y, **select_opts1),
            self.canvas.create_rectangle(omin_x, imax_y,  omax_x, omax_y, **select_opts1),
            # Inner rectangle.
            self.canvas.create_rectangle(imin_x, imin_y,  imax_x, imax_y, **select_opts2)
        )

    def update(self, start, end):
        # Current extrema of inner and outer rectangles.
        imin_x, imin_y,  imax_x, imax_y = self._get_coords(start, end)
        omin_x, omin_y,  omax_x, omax_y = 0, 0,  self.width, self.height

        # Update coords of all rectangles based on these extrema.
        self.canvas.coords(self.rects[0], omin_x, omin_y,  omax_x, imin_y),
        self.canvas.coords(self.rects[1], omin_x, imin_y,  imin_x, imax_y),
        self.canvas.coords(self.rects[2], imax_x, imin_y,  omax_x, imax_y),
        self.canvas.coords(self.rects[3], omin_x, imax_y,  omax_x, omax_y),
        self.canvas.coords(self.rects[4], imin_x, imin_y,  imax_x, imax_y),

        for rect in self.rects:  # Make sure all are now visible.
            self.canvas.itemconfigure(rect, state=tk.NORMAL)

    def _get_coords(self, start, end):
        """ Determine coords of a polygon defined by the start and
            end points one of the diagonals of a rectangular area.
        """
        self.bbox=[min((start[0], end[0])), min((start[1], end[1])),
                max((start[0], end[0])), max((start[1], end[1]))]
        return (min((start[0], end[0])), min((start[1], end[1])),
                max((start[0], end[0])), max((start[1], end[1])))
    def get_coords(self):
        return self.bbox

    def hide(self):
        for rect in self.rects:
            self.canvas.itemconfigure(rect, state=tk.HIDDEN)

class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """

    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise tk.TclError('Cannot use place with the widget ' + self.__class__.__name__)


class CanvasImage:
    """ Display and zoom image """

    def __init__(self, placeholder, path):
        """ Initialize the ImageFrame """
        # a=open("x1.txt","r")
        # data=a.readlines()
        self.image_dirs = []
        self.removed_nodule_list = []
        self.folder = ""
        self.current_slice = 0
        self.imscale = 1.0
        self.app = placeholder
        self.annotations = []
        self.annotated_nodules=[]
        self.count_annotation=0
        self.is_label = False
        self.is_dicom = False
        # self.imscale = 1+(1*int(data[0])/100)  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.3  # zoom magnitude
        self.__filter = Image.ANTIALIAS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        self.__previous_state = 0  # previous state of the keyboard
        self.path = path  # path to the image, should be public for outer classes
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        # Added Option menu
        option_menu = Menu(placeholder)
        placeholder.config(menu=option_menu)
        file_menu = Menu(option_menu)
        option_menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Scan",command=self.open_file) #To_open_file
        file_menu.add_command(label="Open Label", command=self.open_label)  # To_open_file
        file_menu.add_command(label="Exit", command=placeholder.quit)

        hbar.grid(row=0, column=1, sticky='we',columnspan=4)
        vbar.grid(row=1, column=0, sticky='ns',rowspan=4)
        with warnings.catch_warnings():  # suppress DecompressionBombWarning
            warnings.simplefilter('ignore')
            self.__image = Image.open(self.path)  # open image, but down't load it
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set,width=self.imwidth, height=self.imheight)
        self.canvas.grid(row=1, column=1, sticky='nswe', columnspan=12,rowspan=12)
        self.canvas.update()  # wait till canvas is created

        # e=tk.Button(self.__imframe,text="Zoom in",command=self.Zoom,height=3,width=10)
        # e.grid(row=2, column=0)
        f = tk.Button(self.__imframe, text="Zoom out", command=self.Zoom2, height=3, width=10)
        f.grid(row=1, column=14)
        self.l2 = tk.Label(self.__imframe, text="Slice number: ", fg="black", font=26)
        self.l2.grid(row=0, column=14, columnspan=1, sticky=NE)
        #f = tk.Label(self.__imframe, text="Tounch to zoom in", fg="blue", font=26)
        #f.grid(row=3, column=0)

        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas

        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<ButtonPress-2>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B2-Motion>', self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.__wheel)  # zoom for Linux, wheel scroll up
        self.canvas.bind('<Control-r>', self.exclude_annotation)  # zoom for Linux, wheel scroll up
        self.canvas.bind('<Control-R>', self.exclude_annotation)  # zoom for Linux, wheel scroll up

        self.canvas.bind('<Double-Button-1>', self.Zoom)  # zoom for Linux, wheel scroll up
        self.canvas.bind('<Double-Button-3>', self.Zoom2_)  # zoom for Linux, wheel scroll up
        # Handle keystrokes in idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        self.canvas.bind("<Configure>", self.configure)
        # Decide if this image huge or not
        self.__huge = False  # huge or not
        self.__huge_size = 14000  # define size of the huge image
        self.__band_width = 1024  # width of the tile band
        Image.MAX_IMAGE_PIXELS = 1000000000  # suppress DecompressionBombError for the big image

        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size and \
                self.__image.tile[0][0] == 'raw':  # only raw images could be tiled
            self.__huge = True  # image is huge
            self.__offset = self.__image.tile[0][2]  # initial tile offset
            self.__tile = [self.__image.tile[0][0],  # it have to be 'raw'
                           [0, 0, self.imwidth, 0],  # tile extent (a rectangle)
                           self.__offset,
                           self.__image.tile[0][3]]  # list of arguments to the decoder
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side
        # Create image pyramid
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        # Set ratio coefficient for image pyramid
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0  # current image from the pyramid
        self.__scale = self.imscale * self.__ratio  # image pyramide scale
        self.__reduction = 2  # reduction degree of image pyramid
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:  # top pyramid image is around 512 pixels in size
            w /= self.__reduction  # divide on reduction degree
            h /= self.__reduction  # divide on reduction degree
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))

        screen_width = placeholder.winfo_screenwidth()
        screen_height = placeholder.winfo_screenheight()
        print(screen_width,screen_height)
        size = tuple(int(_) for _ in placeholder.geometry().split('+')[0].split('x'))
        print(size)
        #x = screen_width * 0.3 - size[0] * 0.3
        #y = screen_height - size[1]
        x =size[0] * 0.3
        y =0
        self.imagebbox_x = x
        self.imagebbox_y = y
        print(x,y)
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0,0, self.imwidth, self.imheight), width=0)
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas
    def get_canvas(self):
        return self.canvas
    def get_folder(self):
        return self.folder
    def get_currentSlice(self):
        return self.current_slice
    def get_image_dir(self):
        return self.image_dirs;
    def new_annotation(self,new_bbox,current_slice):
        self.count_annotation +=1
        nodule_id = "nodule_"+str(self.count_annotation).rjust(2,'0')
        self.annotated_nodules.append([nodule_id,new_bbox[0],new_bbox[1],new_bbox[2],new_bbox[3],current_slice,self.patient_id])
        return self.annotated_nodules,self.patient_id,self.folder
    def configure(self,event):
        #self.canvas.delete("all")
        #w, h = event.width, event.height
        #self.container = self.canvas.create_rectangle((0, 0,self.imwidth, self.imheight), width=0)
        #print(w)
        #print(h)
        #xy = int(event.width*0.3), 0, w - 1, h - 1
        #self.canvas.create_rectangle(xy)
        #self.canvas.canvasx(int(event.width*0.3))  # get visible area of the canvas
        #self.canvas.canvasy(0)
        #self.canvas.canvasx(w - 1)
        #self.canvas.canvasy(h - 1)
        pass

    def load_lidc_xml(self, agreement_threshold=0, only_patient=None, save_nodules=False):
        pos_lines = []
        with open(self.xml_path, 'r') as xml_file:
            markup = xml_file.read()
        xml = BeautifulSoup(markup, features="xml")
        if xml.LidcReadMessage is None:
            return None, None, None
        patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text

        print("Patient ID: ", patient_id)
        reader = SimpleITK.ImageSeriesReader()
        self.patient_id = patient_id
        original_image = SimpleITK.ReadImage(reader.GetGDCMSeriesFileNames(self.folder, patient_id))
        img_array = SimpleITK.GetArrayFromImage(original_image)
        origin = numpy.array(original_image.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
        spacing = numpy.array(original_image.GetSpacing())  # spacing of voxels in world coor. (mm)
        reading_sessions = xml.LidcReadMessage.find_all("readingSession")

        for reading_session in reading_sessions:
            # print("Sesion")
            nodules = reading_session.find_all("unblindedReadNodule")
            for nodule in nodules:
                nodule_id = nodule.noduleID.text
                # print("  ", nodule.noduleID)
                rois = nodule.find_all("roi")
                x_min = y_min = z_min = 999999
                x_max = y_max = z_max = -999999
                # This line to exclude small nodules (<3mm) for cancer detection
                # if len(rois) < 2:
                #    continue
                nodule_edges_x = []
                nodule_edges_y = []
                for roi in rois:
                    z_pos = float(roi.imageZposition.text)
                    z_min = min(z_min, z_pos)
                    z_max = max(z_max, z_pos)
                    edge_maps = roi.find_all("edgeMap")
                    for edge_map in edge_maps:
                        x = int(edge_map.xCoord.text)
                        y = int(edge_map.yCoord.text)
                        nodule_edges_x.append(x)
                        nodule_edges_y.append(y)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)
                    if x_max == x_min:
                        continue
                    if y_max == y_min:
                        continue
                x_diameter = x_max - x_min
                x_center = x_min + x_diameter / 2
                y_diameter = y_max - y_min
                y_center = y_min + y_diameter / 2
                z_diameter = z_max - z_min
                z_center = z_min + z_diameter / 2
                z_center -= origin[2]
                z_center /= spacing[2]


                line = [nodule_id, x_center, y_center, z_center, x_diameter, y_diameter,patient_id]
                pos_lines.append(line)


        if agreement_threshold > 1:
            filtered_lines = []
            for pos_line1 in pos_lines:
                id1 = pos_line1[0]
                x1 = pos_line1[1]
                y1 = pos_line1[2]
                z1 = pos_line1[3]
                d1 = pos_line1[4]
                overlaps = 0
                for pos_line2 in pos_lines:
                    id2 = pos_line2[0]
                    if id1 == id2:
                        continue
                    x2 = pos_line2[1]
                    y2 = pos_line2[2]
                    z2 = pos_line2[3]
                    d2 = pos_line1[4]
                    dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))
                    if dist < d1 or dist < d2:
                        overlaps += 1
                if overlaps >= agreement_threshold:
                    filtered_lines.append(pos_line1)
                # else:
                #     print("Too few overlaps")
            pos_lines = filtered_lines
        self.annotations = pos_lines
        self.is_label = True
        return pos_lines

    def smaller(self):
        """ Resize image proportionally and return smaller image """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2  # it equals to 1.0
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(w2)  # band length
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1  # compression ratio
            w = int(w2)  # band length
        else:  # aspect_ratio1 < aspect_ration2
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1  # compression ratio
            w = int(h2 * aspect_ratio1)  # band length
        i, j, n = 0, 1, round(0.5 + self.imheight / self.__band_width)
        while i < self.imheight:
            print('\rOpening image: {j} from {n}'.format(j=j, n=n), end='')
            band = min(self.__band_width, self.imheight - i)  # width of the tile band
            self.__tile[1][3] = band  # set band width
            self.__tile[2] = self.__offset + self.imwidth * i * 3  # tile offset (3 bytes per pixel)
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]  # set tile
            cropped = self.__image.crop((0, 0, self.imwidth, band))  # crop tile band
            image.paste(cropped.resize((w, int(band * k) + 1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30 * ' ' + '\r', end='')  # hide printed string
        return image

    def redraw_figures(self):
        """ Dummy function to redraw figures in the children classes """
        pass

    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(1, weight=20)  # make canvas expandable
        self.__imframe.columnconfigure(1, weight=20)

    def pack(self, **kw):
        """ Exception: cannot use pack with this widget """
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        """ Exception: cannot use place with this widget """
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)
    def open_file(self):
        #self.app.filename = filedialog.askopenfilename()
        self.app.directory = filedialog.askdirectory()
        self.folder = self.app.directory
        self._update_folder_label()
        print(self.folder)
        self.read_dicom()
        print(self.image_dirs)
        self.imscale = 1.0
        self.path = self.image_dirs[self.current_slice]
        self.__image.close()
        self.__image = Image.open(self.path)  # reopen / reset image
        self.imwidth, self.imheight = self.__image.size
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        self.__show_image()
        self.l2.config(text="Slice number: " + str(self.current_slice + 1) + "/" + str(len(self.image_dirs)))

    def open_label(self):
        self.app.filename = filedialog.askopenfilename(title = "Select file",filetypes = (("xml files","*.xml"),("all files","*.*")))
        self.xml_path = self.app.filename
        print(self.xml_path)
        pos_annot = self.load_lidc_xml()
        print(pos_annot)
        self.read_dicom(label_image=True,label_list=pos_annot)
    def update_folder_label(self,label):
        self._flabel = label

    def _update_folder_label(self):
        self._flabel.config(text=self.folder)
    # noinspection PyUnusedLocal
    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image

    # noinspection PyUnusedLocal
    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image
    def get_status(self):
        return self.is_dicom,self.is_label
    def exclude_annotation(self,event):
        img_path = "C:/tmp/original/" + "img_" + str(self.current_slice).rjust(4, '0') + "_i.png"
        self.org_img = cv2.imread(img_path)
        exclude_list = []
        for row in self.annotations:
            slice_num = int(round(row[3]))
            if slice_num == self.current_slice:
                exclude_list.append(row)
        for row in exclude_list:
            if row[4] > 0:
                self.xymax = (int(row[1] + row[4] / 2 + 1), int(row[2] + row[5] / 2 + 1))
                self.xymin = (int(row[1] - row[4] / 2 - 1), int(row[2] - row[5] / 2 - 1))
                colorRGB = (0, 255, 0)
            else:
                self.xymax = (int(row[1] + 3 + 1), int(row[2] + 3 + 1))
                self.xymin = (int(row[1] - 3 - 1), int(row[2] - 3 - 1))
                colorRGB = (0, 255, 0)
            tmp_image = copy.copy(self.org_img)
            rect_img = cv2.rectangle(tmp_image, self.xymin, self.xymax, colorRGB, 1)
            cv2.imwrite("C:/tmp/" + "img_" + str(self.current_slice).rjust(4, '0') + "tmp_i.png", rect_img)
            self.path = "C:/tmp/" + "img_" + str(self.current_slice).rjust(4, '0') + "tmp_i.png"
            self.refresh_image()
            self.removeS = 0
            self.canvas.bind('<r>', self.remove_ann)
            self.canvas.bind('<R>', self.remove_ann)
            #self.canvas.bind('<Return>', self.remove_ann)
            var = tk.IntVar()
            button = tk.Button(self.app, text="Next", command=lambda: var.set(1))
            button.place(relx=.3, rely=.3, anchor="c")

            print("waiting...")
            button.wait_variable(var)
            print("done waiting.")
            if self.removeS == 1:
                print("Remove a Nodule with ID: ", row[0])
                self.removed_nodule_list.append(row)
                #colorRGB = (255, 0, 0)
                #org_img = cv2.rectangle(self.org_img, self.xymin, self.xymax, colorRGB, 1)
                #cv2.imwrite("C:/tmp/" + "img_" + str(self.current_slice).rjust(4, '0') + "tmp_i.png", org_img)
                #self.path = "C:/tmp/" + "img_" + str(self.current_slice).rjust(4, '0') + "tmp_i.png"
                #self.refresh_image()
                df = pandas.DataFrame(self.removed_nodule_list,
                                      columns=["nodule_id", "coord_x", "coord_y", "coord_z", "diameter_x", "diameter_y",
                                               "patient_id"])
                df.to_csv("viewer/" +str(row[6]) +"_excluded_annotation_viewer.csv", index=False)
                df.to_csv(self.folder+ "/" +"excluded_annotation_viewer.csv", index=False)
        img_path = "C:/tmp/" + "img_" + str(self.current_slice).rjust(4, '0') + "_i.png"
        cv2.imwrite(img_path, self.org_img)
    def remove_ann(self, event):
        self.removeS = 1
        colorRGB = (255, 0, 0)
        self.org_img = cv2.rectangle(self.org_img, self.xymin, self.xymax, colorRGB, 1)
        cv2.imwrite("C:/tmp/" + "img_" + str(self.current_slice).rjust(4, '0') + "_i.png", self.org_img)
        self.path = "C:/tmp/" + "img_" + str(self.current_slice).rjust(4, '0') + "_i.png"
        self.refresh_image()

    def refresh_image(self):
        self.__image.close()
        self.__image = Image.open(self.path)  # reopen / reset image
        # self.imwidth, self.imheight = self.__image.size
        # self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        # self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()
    def __show_image(self):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        # Vertical part of the image is in the visible area
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            if self.__huge and self.__curr_img < 0:  # show huge image
                h = int((y2 - y1) / self.imscale)  # height of the tile band
                self.__tile[1][3] = h  # set the tile band height
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                self.__image.size = (self.imwidth, h)  # set size of the tile band
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:  # show normal image
                image = self.__pyramid[max(0, self.__curr_img)].crop(  # crop current img from pyramid
                    (int(x1 / self.__scale), int(y1 / self.__scale),
                     int(x2 / self.__scale), int(y2 / self.__scale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                               max(box_canvas[1], box_img_int[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def __move_from(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas

    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down, smaller
            #if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
            #self.imscale /= self.__delta
            #scale /= self.__delta
            if self.current_slice > 0:
                self.current_slice -= 1
                self.path = self.image_dirs[self.current_slice]
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                #self.imwidth, self.imheight = self.__image.size
                #self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
                self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
                self.l2.config(text="Slice number: " + str(self.current_slice + 1) + "/" + str(len(self.image_dirs)))
        if event.num == 4 or event.delta == 120:  # scroll up, bigger
            #i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            #if i < self.imscale: return  # 1 pixel is bigger than the visible area
            #self.imscale *= self.__delta
            #scale *= self.__delta
            if self.current_slice < len(self.image_dirs):
                self.current_slice += 1
                self.path = self.image_dirs[self.current_slice]
                self.__image.close()
                self.__image = Image.open(self.path)  # reopen / reset image
                #self.imwidth, self.imheight = self.__image.size
                #self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
                self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
                self.l2.config(text="Slice number: " + str(self.current_slice + 1) + "/" + str(len(self.image_dirs)))
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()
    def read_dicom(self,label_image=False,label_list=[]):
        reader = SimpleITK.ImageSeriesReader()
        patient_id = os.path.basename(self.folder)
        itk_img = SimpleITK.ReadImage(reader.GetGDCMSeriesFileNames(self.folder, patient_id))
        img_array = SimpleITK.GetArrayFromImage(itk_img)
        #Referash annotations
        self.annotated_nodules = []
        self.count_annotation = 0
        if not os.path.exists("C:/tmp/"):
            os.mkdir("C:/tmp/")
        if not os.path.exists("C:/tmp/original/"):
            os.mkdir("C:/tmp/original/")
        self.image_dirs = []
        for file_path in glob.glob("c:/tmp/*.*"):
            if not os.path.isdir(file_path):
                try:
                    os.remove(file_path)
                except:
                    print("Can't remove file: ", file_path)
        for i in range(img_array.shape[0]):
            img_path = "C:/tmp/original/" + "img_" + str(i).rjust(4, '0') + "_i.png"
            img_path1 = "C:/tmp/" + "img_" + str(i).rjust(4, '0') + "_i.png"
            org_img = img_array[i]
            org_img = helpers.normalize_hu(org_img)
            #org_img = helpers.normalize_hu(org_img)
            if not label_image:
                cv2.imwrite(img_path, org_img * 255)
                self.image_dirs.append(img_path)
            else:
                org_img = cv2.cvtColor(org_img.astype('float32'), cv2.COLOR_GRAY2BGR)
                self.image_dirs.append(img_path1)
                for row in label_list:
                    slice_num = int(round(row[3]))
                    if slice_num == i:
                        if row[4] > 0:
                            xymax = (int(row[1] + row[4] / 2), int(row[2] + row[5] / 2))
                            xymin = (int(row[1] - row[4] / 2), int(row[2] - row[5] / 2))
                            colorRGB = (255, 255, 0)
                        else:
                            xymax = (int(row[1] + 3), int(row[2] + 3))
                            xymin = (int(row[1] - 3), int(row[2] - 3))
                            colorRGB = (0, 0, 255)
                        print("Dicom viewer, slice number: ", str(slice_num).rjust(4, '0'),"\t bbox: ", xymin,xymax)
                        org_img = cv2.rectangle(org_img, xymin, xymax, colorRGB, 1)
                        #break
                cv2.imwrite(img_path1, org_img * 255)
        self.is_dicom = True
    def Zoom(self, event):
        # print("hll")

        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        # x = 10
        # y = 10
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        # if event.num == 5 or event.delta == -120:  # scroll down, smaller
        #    if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
        #     self.imscale /= self.__delta
        #     scale        /= self.__delta
        # if event.num == 4 or event.delta == 120:  # scroll up, bigger
        i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 6
        if i < self.imscale: return  # 1 pixel is bigger than the visible area

        self.imscale *= self.__delta
        scale *= self.__delta
        print(self.imscale)
        #f = open("x1.txt", "w+")
        #f.writelines(str(self.imscale))
        #f.close()

        # if(self.imscale >=12):
        # self.canvas.unbind('<Button-1>')

        # elif(self.imscale <12):
        # self.canvas.bind('<Button-1>',self.Zoom)

        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient

        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))

        #

        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes

        self.__show_image()

    def Zoom2(self):

        """ Zoom with mouse wheel """
        # x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        # y = self.canvas.canvasy(event.y)
        x = 528
        y = 292
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        # if event.num == 5 or event.delta == -120:  # scroll down, smaller
        if round(self.__min_side * self.imscale) < 700: return  # image is less than 30 pixels
        print(round(self.__min_side * self.imscale))
        #     self.imscale /= self.__delta
        #     scale        /= self.__delta
        # if event.num == 4 or event.delta == 120:  # scroll up, bigger
        # i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
        # if i < self.imscale: return  # 1 pixel is bigger than the visible area
        self.imscale /= self.__delta
        scale /= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()
    def Zoom2_(self,event):

        """ Zoom with mouse wheel """
        # x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        # y = self.canvas.canvasy(event.y)
        x = 528
        y = 292
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        # if event.num == 5 or event.delta == -120:  # scroll down, smaller
        if round(self.__min_side * self.imscale) < 700: return  # image is less than 30 pixels
        print(round(self.__min_side * self.imscale))
        #     self.imscale /= self.__delta
        #     scale        /= self.__delta
        # if event.num == 4 or event.delta == 120:  # scroll up, bigger
        # i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
        # if i < self.imscale: return  # 1 pixel is bigger than the visible area
        self.imscale /= self.__delta
        scale /= self.__delta
        # Take appropriate image from the pyramid
        k = self.imscale * self.__ratio  # temporary coefficient
        self.__curr_img = min((-1) * int(math.log(k, self.__reduction)), len(self.__pyramid) - 1)
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        #
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.redraw_figures()  # method for child classes
        self.__show_image()

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right, keys 'd' or 'Right'
                self.__scroll_x('scroll', 1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left, keys 'a' or 'Left'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up, keys 'w' or 'Up'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down, keys 's' or 'Down'
                self.__scroll_y('scroll', 1, 'unit', event=event)

    def crop(self, bbox):
        """ Crop rectangle from the image and return it """
        if self.__huge:  # image is huge and not totally in RAM
            band = bbox[3] - bbox[1]  # width of the tile band
            self.__tile[1][3] = band  # set the tile height
            self.__tile[2] = self.__offset + self.imwidth * bbox[1] * 3  # set offset of the band
            self.__image.close()
            self.__image = Image.open(self.path)  # reopen / reset image
            self.__image.size = (self.imwidth, band)  # set size of the tile band
            self.__image.tile = [self.__tile]
            return self.__image.crop((bbox[0], 0, bbox[2], band))
        else:  # image is totally in RAM
            return self.__pyramid[0].crop(bbox)

    def destroy(self):
        """ ImageFrame destructor """
        self.__image.close()
        map(lambda i: i.close, self.__pyramid)  # close all pyramid images
        del self.__pyramid[:]  # delete pyramid list
        del self.__pyramid  # delete pyramid variable
        self.canvas.destroy()
        self.__imframe.destroy()


class MainWindow(ttk.Frame):
    """ Main window class """

    def __init__(self, mainframe, path):
        """ Initialize the main Frame """
        self.SELECT_OPTS = dict(dash=(2, 2), stipple='gray25', fill='red',
                           outline='')
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Radiologics Medical Dicom Viewer 0.5')
        # self.master.geometry('800x450+100+10')  # size of the main window
        self.master.geometry('920x600')  # size of the main window
        self.master.rowconfigure(1, weight=20)  # make the CanvasImage widget expandable
        self.master.columnconfigure(1, weight=20)
        self.canvas_class = CanvasImage(self.master, path)  # create widget
        self.canvas_class.grid(row=1, column=1)  # show widget
        f = tk.Button(self.master, text="Annotation Mode", command=self.annotation, height=3, width=20)
        f.grid(row=5, column=1)
        scan_folder = self.canvas_class.get_folder()
        if scan_folder != "":
            fldr = scan_folder
        else:
            fldr = "No dicom file loaded"
        l = tk.Label(self.master, text=fldr, fg="black", font=10)
        self.canvas_class.update_folder_label(l)
        l.grid(row=0, column=0,columnspan=6,sticky=W)

        self.master.rowconfigure(0, weight=1)  # make canvas expandable
        self.master.columnconfigure(0, weight=1)
        self.canvas = self.canvas_class.get_canvas()

    def annotation(self):
        loaded_dicom,loaded_labels = self.canvas_class.get_status()
        if loaded_dicom:
            self.current_slice = self.canvas_class.get_currentSlice()
            if loaded_labels:
                self.path = "C:/tmp/" + "img_" + str(self.current_slice).rjust(4, '0') + "_i.png"
            else:
                self.path = "C:/tmp/original/" + "img_" + str(self.current_slice).rjust(4, '0') + "_i.png"
            print("Annotater view slice number: ", str(self.current_slice).rjust(4, '0'))
            self.newWindow = Toplevel(self.master)
            self.__image = Image.open(self.path)
            img = ImageTk.PhotoImage(self.__image)
            self.canvas = tk.Canvas(self.newWindow, width=img.width(), height=img.height(),
                                    borderwidth=0, highlightthickness=0)
            self.canvas.pack(expand=True)

            self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
            self.canvas.img = img  # Keep reference.
            # sets the title of the
            # Toplevel widget
            self.newWindow.title("Annotater")

            # sets the geometry of toplevel
            self.newWindow.geometry("920x600")
            self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
            self.selection_obj = SelectionObject(self.canvas, self.SELECT_OPTS)
            f = tk.Button(self.newWindow, text="Save Annotation", command=self.save_annotation, height=3, width=20)
            f.pack()
            l = tk.Label(self.newWindow, text="Click Save Annotation to save the bbox as csv.", fg="black", font=26)
            l.pack()
            # Callback function to update it given two points of its diagonal.
            def on_drag(start, end, **kwarg):  # Must accept these arguments.
                self.selection_obj.update(start, end)

            # Create mouse position tracker that uses the function.
            self.posn_tracker = MousePositionTracker(self.canvas)
            self.posn_tracker.autodraw(command=on_drag)  # Enable callbacks.
        else:
            messagebox.showerror(title="No dicom file loaded", message="Load dicom file to enable this feature ")
    def save_annotation(self):
        print(self.selection_obj.get_coords())
        annotated_nodules_list,patient_id,folder = self.canvas_class.new_annotation(self.selection_obj.get_coords(),self.current_slice)
        df = pandas.DataFrame(annotated_nodules_list,
                              columns=["nodule_id", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_x1", "slice_num",
                                       "patient_id"])
        df.to_csv("viewer/" + str(patient_id) + "_new_annotation_viewer.csv", index=False)
        df.to_csv(folder + "/" + "new_annotation_viewer.csv", index=False)

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        image_dirs = self.canvas_class.get_image_dir()
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        def on_drag(start, end, **kwarg):  # Must accept these arguments.
            self.selection_obj.update(start, end)
        if event.num == 5 or event.delta == -120:  # scroll down, smaller
            if self.current_slice > 0:
                self.current_slice -= 1
                self.path = image_dirs[self.current_slice]
                self.__image.close()
                img = ImageTk.PhotoImage(Image.open(self.path))
                self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
                self.canvas.img = img  # Keep reference.
                self.selection_obj = SelectionObject(self.canvas, self.SELECT_OPTS)
                self.posn_tracker = MousePositionTracker(self.canvas)
                self.posn_tracker.autodraw(command=on_drag)  # Enable callbacks.
        if event.num == 4 or event.delta == 120:  # scroll up, bigger
            #i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            #if i < self.imscale: return  # 1 pixel is bigger than the visible area
            #self.imscale *= self.__delta
            #scale *= self.__delta
            if self.current_slice < len(image_dirs):
                self.current_slice += 1
                self.path = image_dirs[self.current_slice]
                self.__image.close()
                img = ImageTk.PhotoImage(Image.open(self.path))
                self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
                self.canvas.img = img  # Keep reference.
                self.selection_obj = SelectionObject(self.canvas, self.SELECT_OPTS)
                self.posn_tracker = MousePositionTracker(self.canvas)
                self.posn_tracker.autodraw(command=on_drag)  # Enable callbacks.


image_width = int(2* 920/3)
image_height = int(2* 600/3)
img = Image.new('RGB', (image_width, image_height), color=(139,0,0))
# create the canvas
first_image = ImageDraw.Draw(img)
font = ImageFont.truetype("arial.ttf", size=48)
text_width, text_height = first_image.textsize('Radiologics Medical', font=font)
print(f"Text width: {text_width}")
print(f"Text height: {text_height}")
x_pos = int((image_width - text_width) / 2)
y_pos = int((image_height - text_height) / 2)
first_image.text((x_pos, y_pos), "Radiologics Medical", font=font, fill='#FFFFFF')

if not os.path.exists("C:/tmp/"):
    os.mkdir("C:/tmp/")
if not os.path.exists("C:/tmp/first/"):
    os.mkdir("C:/tmp/first/")
img.save("C:/tmp/first/radiologicsmedical.png","PNG")
filename = "C:/tmp/first/radiologicsmedical.png"  # place path to your image here

if not os.path.exists("viewer/"):
        os.mkdir("viewer/")
app = MainWindow(tk.Tk(), path=filename)




app.mainloop()