import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from threading import Thread
from os.path import splitext
from scipy import fft, signal
from time import sleep, perf_counter
from themes import Tk, Frame, Label, Entry, Button, OptionMenu
from camera import DummyCam, ZL41Wave, SK2048U3, TCE1304U


class Updater(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera = None
        self.cameras = {}

        for camera in [ZL41Wave, SK2048U3, TCE1304U]:
            try:
                self.cameras[camera.__name__] = camera()
            except: pass

        self.cameras['2048x6.5um']  = DummyCam(2048, 6.5)
        self.cameras['2048x14um']  = DummyCam(2048, 14)
        self.cameras['3648x8.0um'] = DummyCam(3648, 8)

        self.raw_data = []
        self.fft_data = []
        self.paused = True
        self.running = True

    def close(self):
        for camera in self.cameras.values():
            camera.close_camera()

    def set_dummy_signal(self, *peaks, fwhm=0.5):
        for camera in self.cameras.values():
            if camera.is_dummy:
                camera.set_dummy_signal(*peaks, fwhm=fwhm)

    def set_camera(self, camera, camera_gain, exposure_time):
        self.camera = camera
        self.camera.set_camera_gain(camera_gain)
        self.camera.set_exposure_time(exposure_time)

    def set_handler(self, image_handler, spectrum_handler, accum_count, pixel_count):
        self.image_handler = image_handler
        self.spectrum_handler = spectrum_handler
        self.raw_data = np.zeros((accum_count, pixel_count))
        self.fft_data = np.zeros((accum_count, pixel_count*4))

    def run(self):
        def cut_below(data, λ_min):
            data_fft = fft.rfft(data)
            data_fft[:int(len(data)/(2*λ_min))] = 0
            data_ifft = np.real(fft.irfft(data_fft))
            return data_ifft

        def get_fft(data):
            return 7.5 * np.abs(fft.fft(np.pad(data, (0, 7*len(data))))[1:1+4*len(data)]) / len(data)

        n = 0
        while self.running:
            if self.paused or not self.camera:
                sleep(0.01)
                continue

            raw_data = self.camera.get_frame()
            if len(raw_data.shape) == 1:
                fft_data = get_fft(raw_data)

                if n >= len(self.raw_data): n = 0
                if len(raw_data) == len(self.raw_data[n]):
                    self.raw_data[n] = raw_data
                    self.fft_data[n] = fft_data
                    y1 = np.average(self.raw_data, axis=0)
                    y2 = np.average(self.fft_data, axis=0)
                    self.spectrum_handler(y1, y2)
                    n += 1
            else:
                self.image_handler(raw_data)


class Image(FigureCanvasTkAgg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ax = self.figure.add_subplot()
        self.image = self.ax.imshow(np.random.rand(2048,2048))

    def show(self, image):
        self.image.set_data(image)
        self.draw()


class Plotter(FigureCanvasTkAgg):
    def _inv(self, λ): return np.divide(1, λ, where=λ!=0)
    def _raman(self, λ): return 1e7 * (1 / self.λ_0 - self._inv(λ))
    def _invraman(self, Δ): return self._inv(1 / self.λ_0 - Δ / 1e7)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logscale = False
        self.pixel_count = 2
        self.pixel_pitch = 14336
        self.λ_min = 1
        self.λ_0 = 1

        self.ax1 = self.figure.add_subplot(313)
        self.ax2 = self.figure.add_subplot(312)
        self.ax3 = self.figure.add_subplot(311)
        self.ax4 = self.ax3.secondary_xaxis('bottom', functions=(self._invraman, self._raman))

        self.ax1.set_xlabel('Sensor Position (mm)')
        self.ax2.set_xscale('function', functions=(self._inv, self._inv))
        self.ax3.set_xlabel('Raman Shift (cm⁻¹)')

        self.ax1.set_xticks(np.arange(-14, 15, 1))
        self.ax2.set_xticks(1/np.linspace(1/400, 1/12000, 30))
        self.ax3.set_xticks(np.arange(-1200, 1201, 100))
        self.ax3.set_xlim(-1200, 1200)

        self.ax3.xaxis.tick_top()
        self.ax3.xaxis.set_label_position('top')
        self.ax4.minorticks_on()

        self.ax1.grid()
        self.ax2.grid()
        self.ax3.grid()

        self.line1, = self.ax1.plot((0, 1), (1, 1), color='#0F4', animated=True)
        self.line2, = self.ax2.plot((0, 1), (1, 1), color='#F80', animated=True)
        self.line3, = self.ax3.plot((0, 1), (1, 1), color='#F80', animated=True)

        self.ax1.add_line(self.line1)
        self.ax2.add_line(self.line2)
        self.ax3.add_line(self.line3)

        self.draw()
        self.background = self.copy_from_bbox(self.figure.bbox)

    def set_axes(self, λ_min=None, λ_0=None, camera=None):
        if λ_min: self.λ_min = λ_min
        if λ_0:   self.λ_0   = λ_0

        if camera:
            self.pixel_count = camera.pixel_count
            self.pixel_pitch = camera.pixel_pitch

        if self.logscale:
            self.ax2.set_yscale('log')
            self.ax3.set_yscale('log')
            self.ax2.set_ylim(1e-6, 1)
            self.ax3.set_ylim(1e-6, 1)
        else:
            self.ax2.set_yscale('linear')
            self.ax3.set_yscale('linear')
            self.ax2.set_ylim(-0.01, 1.01)
            self.ax3.set_ylim(-0.01, 1.01)

        self.ax1.set_ylim(-0.01, 1.01)
        self.ax1.set_xlim(-self.pixel_pitch * self.pixel_count / 2000, self.pixel_pitch * self.pixel_count / 2000)
        self.ax2.set_xlim(1e7, max(400, self.λ_min))
        self.ax4.set_xticks(np.arange(300, 1220, 2 if self.λ_0 < 600 else 5 if self.λ_0 < 900 else 10))

        self.draw()
        self.background = self.copy_from_bbox(self.figure.bbox)

    def auto_scale(self, *args):
        ymax = min(1.2 * max(np.max(np.abs(self.line1.get_ydata())), 1e-3), 1)
        self.ax1.set_ylim(-0.01 * ymax, 1.01 * ymax)

        data = self.line2.get_ydata()
        ymax = max(np.max(data[len(data)//2:]), 1e-3)

        if self.logscale:
            ymin = max(np.min(data[len(data)//2:]), 1e-7)
            self.ax2.set_ylim(ymin, ymax**1.1 / ymin**0.1)
            self.ax3.set_ylim(ymin, ymax**1.1 / ymin**0.1)

        else:
            self.ax2.set_ylim(-0.012 * ymax, 1.2 * ymax)
            self.ax3.set_ylim(-0.012 * ymax, 1.2 * ymax)

        self.draw()
        self.background = self.copy_from_bbox(self.figure.bbox)
        self.ax1.draw_artist(self.line1)
        self.ax2.draw_artist(self.line2)
        self.ax3.draw_artist(self.line3)
        self.blit(self.figure.bbox)

    def set_data(self, y1, y2):
        x1 = (0.5 + np.arange(-len(y1)//2, len(y1)//2)) * self.pixel_pitch / 1000
        x2 = 1/np.linspace(1/(len(y2)*self.λ_min), 2/self.λ_min, 2*len(y2))[:len(y2)]
        x3 = self._raman(x2)
        self.line1.set_data(x1, y1)
        self.line2.set_data(x2, y2)
        self.line3.set_data(x3, y2)

        self.restore_region(self.background)
        self.ax1.draw_artist(self.line1)
        self.ax2.draw_artist(self.line2)
        self.ax3.draw_artist(self.line3)
        self.blit(self.figure.bbox)

    def get_data(self):
        x1 = self.line1.get_xdata()
        y1 = self.line1.get_ydata()
        x2 = self.line2.get_xdata()[::-1]
        y2 = self.line2.get_ydata()[::-1]
        return x1, y1, x2, y2


class App(Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title('Multi-channel Fourier Transform Raman Spectrometer')
        self.protocol('WM_DELETE_WINDOW', self.quit)

        self.image = Image(Figure(), self)
        self.plotter = Plotter(Figure(), self)
        self.plotter.get_tk_widget().pack(side='top')
        self.updater = Updater()
        self.updater.start()
        self.controls = Frame(self)
        self.controls.pack(fill='x', side='bottom')
        self.aoi_controls = Frame(self.controls)
        self.dummy_controls = Frame(self.controls)

        self.camera_type = tk.StringVar(self)
        self.accum_count = tk.IntVar(self, 1)
        self.camera_gain = tk.DoubleVar(self, 20)
        self.exposure_time = tk.DoubleVar(self, 1)
        self.λ_min = tk.DoubleVar(self, 500)
        self.λ_0   = tk.DoubleVar(self, 532)
        self.dummy_signals = [tk.DoubleVar(self, x) for x in (0.01, 518, 1, 532, 0.1, 547, 0.4)]

        camera_type = OptionMenu(self.controls, self.camera_type, *self.updater.cameras.keys())
        accum_count = Entry(self.controls, textvariable=self.accum_count)
        camera_gain = Entry(self.controls, textvariable=self.camera_gain)
        exposure_time = Entry(self.controls, textvariable=self.exposure_time)
        λ_min = Entry(self.controls, textvariable=self.λ_min)
        λ_0 = Entry(self.controls, textvariable=self.λ_0)

        Label(self.controls, text=' Camera:').pack(side='left')
        camera_type.pack(side='left', pady=3)
        Label(self.controls, text='  Accum =').pack(side='left')
        accum_count.pack(side='left')
        Label(self.controls, text='  Gain =').pack(side='left')
        camera_gain.pack(side='left')
        Label(self.controls, text='  Exposure =').pack(side='left')
        exposure_time.pack(side='left')
        Label(self.controls, text='ms,    Spectrum:  λₘᵢₙ =').pack(side='left')
        λ_min.pack(side='left')
        Label(self.controls, text='nm  λ₀ =').pack(side='left')
        λ_0.pack(side='left')
        Label(self.controls, text='nm').pack(side='left')

        self.camera_type.trace('w', self.select_camera)

        self.buttons = []
        self.buttons.append(Button(self.controls, text='Auto scale', command=self.plotter.auto_scale))
        self.buttons.append(Button(self.controls, text='Log/Linear', command=self.toggle_logscale))
        self.buttons.append(Button(self.controls, text='Save as...', command=self.save_plot))
        self.buttons.append(Button(self.controls, text='Quit', command=self.quit))
        for button in reversed(self.buttons):
            button.pack(padx=2, pady=3, side='right')
        Label(self.controls, text='  ').pack(side='right')

        self.aoitop = tk.IntVar(self, 1008)
        self.aoivbin = tk.IntVar(self, 32)
        Label(self.aoi_controls, text='Area of Interest: ').pack(side='left')
        Label(self.aoi_controls, text='y₀ =').pack(side='left')
        aoitop = Entry(self.aoi_controls, textvariable=self.aoitop)
        aoitop.pack(side='left')
        Label(self.aoi_controls, text=' Δy =').pack(side='left')
        aoivbin = Entry(self.aoi_controls, textvariable=self.aoivbin)
        aoivbin.pack(side='left')
        Label(self.aoi_controls, text='  ').pack(side='left')
        Button(self.aoi_controls, text='Image', command=self.show_image).pack(padx=2, pady=3, side='left')
        Button(self.aoi_controls, text='Spectrum', command=self.show_spectrum).pack(padx=2, pady=3, side='left')

        Label(self.dummy_controls, text='Dummy:').pack(side='left')
        dummy_signals = [Entry(self.dummy_controls, textvariable=x) for x in self.dummy_signals]
        for text, widget in zip(['A₁', 'λ₁', 'A₂', 'λ₂', 'A₃', 'λ₃', 'FWHM'], dummy_signals):
            Label(self.dummy_controls, text=f' {text} =').pack(side='left')
            widget.pack(side='left')

        for event in ['<Return>', '<FocusOut>']:
            accum_count.bind(event, self.set_accum_count)
            camera_gain.bind(event, self.set_camera_gain)
            exposure_time.bind(event, self.set_exposure_time)
            λ_min.bind(event, self.set_axes)
            λ_0.bind(event, self.set_axes)
            aoitop.bind(event, self.set_area_of_interest)
            aoivbin.bind(event, self.set_area_of_interest)
            for widget in dummy_signals:
                widget.bind(event, self.set_dummy_signal)

        self.set_axes(self.λ_min.get(), self.λ_0.get())
        self.set_dummy_signal()
        self.bind('<Control-a>', self.plotter.auto_scale)
        self.bind('<Control-l>', self.toggle_logscale)
        self.bind('<Control-s>', self.save_plot)
        self.bind('<Control-q>', self.quit)

    def show_image(self):
        self.plotter.get_tk_widget().forget()
        self.updater.camera.set_area_of_interest(1, 0, 2048)
        self.image.get_tk_widget().pack(side='top')

    def show_spectrum(self):
        self.image.get_tk_widget().forget()
        self.updater.camera.set_area_of_interest(self.aoivbin.get(), self.aoitop.get(), 1)
        self.plotter.get_tk_widget().pack(side='top')

    def select_camera(self, *args):
        self.updater.paused = True
        camera = self.updater.cameras[self.camera_type.get()]
        if camera.is_dummy:
            self.aoi_controls.forget()
            self.dummy_controls.pack(side='right')
        elif camera.cam:
            self.dummy_controls.forget()
            self.aoi_controls.pack(side='right')
        else:
            self.aoi_controls.forget()
            self.dummy_controls.forget()
        self.title(self.title().split(' - ')[0] + ' - ' + str(camera))
        self.set_accum_count()
        self.plotter.set_axes(self.λ_min.get(), self.λ_0.get(), camera)
        self.updater.set_camera(camera, self.camera_gain.get(), self.exposure_time.get())
        self.updater.paused = False

    def set_accum_count(self, *args):
        try:
            accum_count = self.accum_count.get()
            camera = self.updater.cameras.get(self.camera_type.get())
            if camera != self.updater.camera or accum_count > 0 and accum_count != len(self.updater.raw_data):
                if camera:
                    self.updater.set_handler(self.image.show, self.plotter.set_data, accum_count, camera.pixel_count)
        except tk.TclError: pass

    def set_camera_gain(self, *args):
        try:
            camera_gain = self.camera_gain.get()
            if self.updater.camera and camera_gain != self.updater.camera.camera_gain:
                self.updater.camera.set_camera_gain(self.camera_gain.get())
        except tk.TclError: pass

    def set_exposure_time(self, *args):
        try:
            exposure_time = self.exposure_time.get()
            if self.updater.camera and exposure_time != self.updater.camera.exposure_time:
                self.updater.camera.set_exposure_time(self.exposure_time.get())
        except tk.TclError: pass

    def set_axes(self, event, *args):
        try:
            if self.λ_min.get() != self.plotter.λ_min or self.λ_0.get() != self.plotter.λ_0:
                if self.λ_min.get() > 0 and self.λ_0.get() > 0 and self.λ_min.get() < self.λ_0.get():
                    self.set_dummy_signal()
                    self.plotter.set_axes(self.λ_min.get(), self.λ_0.get())
        except tk.TclError: pass

    def set_area_of_interest(self, *args):
        try:
            self.updater.camera.set_area_of_interest(self.aoivbin.get(), self.aoitop.get(), 1)
        except tk.TclError: pass

    def set_dummy_signal(self, *args):
        try:
            s = [s.get() for s in self.dummy_signals]
            self.updater.set_dummy_signal(self.λ_min.get(), (s[0], s[1]), (s[2], s[3]), (s[4], s[5]), fwhm=s[6])
        except tk.TclError: pass

    def toggle_logscale(self, *args):
        self.plotter.logscale = not self.plotter.logscale
        self.plotter.set_axes()

    def save_plot(self, *args):
        self.updater.paused = True

        if self.updater.camera:
            filetypes = [('All files: *.csv, *.png, *.svg', '*.csv *.png *.svg'),
                         ('Comma separated values files: *.csv', '*.csv'),
                         ('Portable network graphics files: *.png', '*.png'),
                         ('Scalable vector graphics files: *.svg', '*.svg')]
            filename = tk.filedialog.asksaveasfilename(filetypes=filetypes)

            if filename:
                filename = splitext(filename)
                if not filename[1] or filename[1] == '.csv':
                    with open(filename[0]+'.csv', 'w') as f:
                        f.write(f'λ (nm), intensity, x (mm), intensity\n')
                        x1, y1, x2, y2 = self.plotter.get_data()
                        for i in range(len(x1)):
                            f.write(f'{x2[i]},{y2[i]},{x1[i]},{y1[i]}\n')
                        for i in range(len(x1), len(x2)):
                            f.write(f'{x2[i]},{y2[i]}\n')

                if not filename[1] or filename[1] == '.png':
                    self.plotter.figure.savefig(filename[0]+'.png')

                if not filename[1] or filename[1] == '.svg':
                    self.plotter.figure.savefig(filename[0]+'.svg')

        self.updater.paused = False

    def quit(self, *args):
        if self.updater.paused:
            self.updater.running = False
            self.updater.join(0.1)
            if self.updater.is_alive():
                self.after(100, self.quit)
            else:
                self.updater.close()
                super().quit()
        else:
            self.updater.paused = True
            if tk.messagebox.askyesno('McFT Raman Spectrometer', 'Do you want to close the application?'):
                self.after(100, self.quit)
            else:
                self.updater.paused = False


if __name__ == '__main__':
    app = App()
    app.geometry("1600x993")
    app.mainloop()
