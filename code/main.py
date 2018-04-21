import tkinter as tk
import os.path
from threading import Timer


import h5py
import numpy as np
from PIL import Image, ImageTk

from tracker import Tracker


class App(tk.Frame):
    def __init__(self, master, player_detection_file, line_annotation_file, video):
        """
        Method generates and initiates the Graphical interface for the display.
        :param master: The master Tkinter element.
        :param player_detection_file: The player annotation file
        :param line_annotation_file: The line annotation file
        :param video: The video referenced by line_annos and plaer_dets
        """
        tk.Frame.__init__(self, master)
        self.pack(side=tk.LEFT)
        self.master.title("Rugby Tracker")
        self.master.resizable(False, False)
        self.tracker = Tracker(video, player_detection_file, line_annotation_file)

        self.lines = False
        self.footage_detections = False
        self.homographies = False
        self.filters = True
        self.should_play = False

        # Define the footage
        self.footage_img = Image.new('RGB', (1280, 720))
        self.footage_img_tk = ImageTk.PhotoImage(self.footage_img)
        self.footage_label = tk.Label(self, image=self.footage_img_tk)
        self.footage_label.grid(column=0, row=0, rowspan=4)

        # Define the pitch
        self.pitch_img = Image.new('RGB', (600, 350))
        self.pitch_img_tk = ImageTk.PhotoImage(self.pitch_img)
        self.pitch_label = tk.Label(self, image=self.pitch_img_tk)
        self.pitch_label.grid(column=1, row=0, sticky='N', columnspan=2)

        # Define the options menu
        self.options_menu = tk.Frame(self)
        self.options_menu.grid(column=1, row=1, sticky='nesw', padx=10, rowspan=2)
        tk.Label(self.options_menu, text='Toggles:', font='Helvetica 18 bold').pack(anchor='n')
        self.lines_but = tk.Button(self.options_menu, text='Footage Lines [Off]', font='Helvetica 14',
                                   command=self.toggle_lines)
        self.lines_but.pack(fill='both', anchor='w')
        self.footage_but = tk.Button(self.options_menu, text='Footage Detections [Off]', font='Helvetica 14',
                                     command=self.toggle_footage_detections)
        self.footage_but.pack(fill='both', anchor='w')
        self.homog_but = tk.Button(self.options_menu, text='Homography translations [Off]', font='Helvetica 14',
                                   command=self.toggle_homographies)
        self.homog_but.pack(fill='both', anchor='w')
        self.filter_but = tk.Button(self.options_menu, text='Particle Filters [On]', font='Helvetica 14',
                                    command=self.toggle_particles)
        self.filter_but.pack(fill='both', anchor='w')

        self.changedLabel = tk.Label(self, text='SCENE CHANGED', fg='red', font=("Courier", 38))

        # Homography display
        self.homography_frame = tk.Label(self)
        self.homography_frame.grid(column=2, row=1, sticky='new')
        tk.Label(self.homography_frame, text='Homography Matrix:', font='Helvetica 18 bold').pack(anchor='n')
        self.homography_text = tk.Label(self.homography_frame, text='')
        self.homography_text.pack()

        self.players_and_frames = tk.Label(self, text='Number of active filters: \nFrame:', font='Helvetica 14',
                                           justify='left')
        self.players_and_frames.grid(column=2, row=2, sticky='new')

        # Pause/Play button
        self.pp_but = tk.Button(self, text=' > Play > ', font='Helvetica 18 bold', command=self.toggle_play)
        self.pp_but.grid(column=1, row=3, sticky='ew', padx=10, columnspan=2)

        # For the homography, set precision to be smaller
        np.set_printoptions(precision=4)

    def toggle_particles(self):
        """
        Method toggles on or off the displaying of the particle filter outputs.
        """
        self.filters = not self.filters
        self.filter_but.configure(text='Partticle Filters [' + ('On' if self.filters else 'Off') + ']')

    def toggle_lines(self):
        """
        Method toggles on or off the displaying of the line annotations.
        """
        self.lines = not self.lines
        self.lines_but.configure(text='Footage Lines [' + ('On' if self.lines else 'Off') + ']')

    def toggle_footage_detections(self):
        """
        Method toggles on or off the displaying of the player detections.
        """
        self.footage_detections = not self.footage_detections
        self.footage_but.configure(text='Footage Detections [' + ('On' if self.footage_detections else 'Off') + ']')

    def toggle_homographies(self):
        """
        Method toggles on or off the displaying of the player detections after the Homography transform.
        """
        self.homographies = not self.homographies
        self.homog_but.configure(text='Homography translations [' + ('On' if self.homographies else 'Off') + ']')

    def toggle_play(self):
        """
        Method pauses or plays the video.
        """
        self.should_play = not self.should_play
        self.pp_but.configure(text=' || Pause || ' if self.should_play else ' > Play > ')
        self.master.after(1, self.update_frames)

    def _stop_change_label(self):
        """
        Method used to hide the 'SCENE CHANGED' label when called.
        """
        self.changedLabel.grid_remove()

    def update_frames(self):
        """
        Main method of the graphical interface, interacting with the Tracker object & displaying the required outputs.
        """
        if self.should_play:
            pitch, footage, changed, H, n_filters, frame_no = \
                self.tracker.get_frame(self.lines, self.footage_detections, self.homographies, self.filters)
            self.pitch_img = Image.fromarray(pitch)
            self.pitch_img_tk = ImageTk.PhotoImage(image=self.pitch_img)
            self.pitch_label.configure(image=self.pitch_img_tk)

            self.players_and_frames.configure(
                text='Number of active filters: ' + str(n_filters) + '\nFrame: ' + str(frame_no))

            if H is not None:
                H = [v for v in np.nditer(H)]
                string = ""
                for i in range(3):
                    string += "{0: >+8f}{3: >4}{1: >+8f}{3: >4}{2: >+8f} \n".format(*H[i * 3:(i * 3) + 3], '')
                self.homography_text.configure(text=string)

            self.footage_img = Image.fromarray(footage)
            self.footage_img_tk = ImageTk.PhotoImage(image=self.footage_img)
            self.footage_label.configure(image=self.footage_img_tk)

            if changed:
                self.changedLabel.grid(column=0, row=0, sticky='nw')
                Timer(3.0, self._stop_change_label).start()

            self.master.after(1, self.update_frames)


if __name__ == '__main__':
    # Start the graphical interface with the required files
    root = tk.Tk()

    dirname = os.path.dirname(__file__)
    with h5py.File(os.path.join(dirname, '../demonstration_data/line_annotations.h5'), 'r') as line_annos:
        with h5py.File(os.path.join(dirname, '../demonstration_data/player_annotations.h5'), 'r') as player_dets:
            app = App(root, player_dets, line_annos, os.path.join(dirname, '../demonstration_data/footage.mp4'))
            app.mainloop()
