import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# Import scipy.signal for filtering
from scipy.signal import butter, lfilter

class RFSignalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RF Signal Processing GUI")

        # Default Parameters
        self.f_signal = 5e6        # Signal frequency (Hz)
        self.f_carrier = 21e6      # Carrier frequency (Hz)
        self.fs = 50e6             # Sampling rate (Hz)
        self.data_width = 12       # Data width (bits)
        self.duration = 1e-6       # Signal duration (seconds)
        self.N_fft = 1024          # Number of FFT points
        self.butter = 5            # Butterworth Filter Order
        self.lowcut = 0.0          # Low end cutoff frequency for Butterworth Filter
        self.highcut = 5e6         # High end cutoff frequency for Butterworth Filter

        # List to keep track of open plot windows
        self.plot_windows = []

        # Create Main GUI
        self.create_main_gui()

    def create_main_gui(self):
        # Main Parameters Frame
        params_frame = tk.Frame(self.root)
        params_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=10, pady=10)

        # General Parameters Frame
        general_params_frame = tk.LabelFrame(params_frame, text="General Parameters")
        general_params_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)

        # General Parameters
        general_params = [
            ("Signal Frequency (Hz)", self.f_signal),
            ("Carrier Frequency (Hz)", self.f_carrier),
            ("Sampling Rate (Hz)", self.fs),
            ("Data Width (bits)", self.data_width),
            ("Signal Duration (s)", self.duration),
        ]

        self.param_vars = {}

        for idx, (label_text, default_value) in enumerate(general_params):
            label = tk.Label(general_params_frame, text=label_text)
            label.grid(row=idx, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.StringVar(value=str(default_value))
            entry = tk.Entry(general_params_frame, textvariable=var)
            entry.grid(row=idx, column=1, padx=5, pady=5)
            self.param_vars[label_text] = var

        # Plot-Specific Parameters Frame
        plot_params_frame = tk.LabelFrame(params_frame, text="Plot-Specific Parameters")
        plot_params_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)

        # FFT Parameters Frame
        fft_params_frame = tk.LabelFrame(plot_params_frame, text="FFT Analysis Parameters")
        fft_params_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)

        fft_params = [
            ("Number of FFT Points", self.N_fft),
        ]

        for idx, (label_text, default_value) in enumerate(fft_params):
            label = tk.Label(fft_params_frame, text=label_text)
            label.grid(row=idx, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.StringVar(value=str(default_value))
            entry = tk.Entry(fft_params_frame, textvariable=var)
            entry.grid(row=idx, column=1, padx=5, pady=5)
            self.param_vars[label_text] = var

        # I/Q Parameters Frame
        iq_params_frame = tk.LabelFrame(plot_params_frame, text="I/Q Components Parameters")
        iq_params_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)

        iq_params = [
            ("Butterworth Filter Order", self.butter),
            ("Butterworth Low End Cutoff Frequency", self.lowcut),
            ("Butterworth High End Cutoff Frequency", self.highcut),
        ]

        for idx, (label_text, default_value) in enumerate(iq_params):
            label = tk.Label(iq_params_frame, text=label_text)
            label.grid(row=idx, column=0, sticky=tk.W, padx=5, pady=5)
            var = tk.StringVar(value=str(default_value))
            entry = tk.Entry(iq_params_frame, textvariable=var)
            entry.grid(row=idx, column=1, padx=5, pady=5)
            self.param_vars[label_text] = var

        # Update Button
        update_button = tk.Button(params_frame, text="Update Parameters", command=self.update_parameters)
        update_button.pack(side=tk.TOP, pady=10)

        # Create Menu
        self.create_menu()

    def create_menu(self):
        menubar = tk.Menu(self.root)

        # Graph Menu
        graph_menu = tk.Menu(menubar, tearoff=0)
        graph_menu.add_command(label="Analog Signal", command=lambda: self.open_plot_window("Analog Signal"))
        graph_menu.add_command(label="Sampled and Quantized Signal", command=lambda: self.open_plot_window("Sampled and Quantized Signal"))
        graph_menu.add_command(label="FFT Analysis", command=lambda: self.open_plot_window("FFT Analysis"))
        graph_menu.add_command(label="I and Q Components", command=lambda: self.open_plot_window("I and Q Components"))
        graph_menu.add_command(label="Amplitude and Phase", command=lambda: self.open_plot_window("Amplitude and Phase"))
        menubar.add_cascade(label="Graphs", menu=graph_menu)

        # Options Menu
        options_menu = tk.Menu(menubar, tearoff=0)
        options_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="Options", menu=options_menu)

        self.root.config(menu=menubar)

    def update_parameters(self):
        try:
            self.f_signal = float(self.param_vars["Signal Frequency (Hz)"].get())
            self.f_carrier = float(self.param_vars["Carrier Frequency (Hz)"].get())
            self.fs = float(self.param_vars["Sampling Rate (Hz)"].get())
            self.data_width = int(self.param_vars["Data Width (bits)"].get())
            self.duration = float(self.param_vars["Signal Duration (s)"].get())
            self.N_fft = int(self.param_vars["Number of FFT Points"].get())
            self.butter = int(self.param_vars["Butterworth Filter Order"].get())
            self.lowend = float(self.param_vars["Butterworth Low End Cutoff Frequency"].get())
            self.highend = float(self.param_vars["Butterworth High End Cutoff Frequency"].get())

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
            return

        # Update all open plot windows
        for window in self.plot_windows:
            window.update_plot()

    def open_plot_window(self, plot_type):
        plot_window = PlotWindow(self, plot_type)
        self.plot_windows.append(plot_window)
        # Remove the window from the list when it is closed
        plot_window.window.protocol("WM_DELETE_WINDOW", lambda w=plot_window: self.close_plot_window(w))

    def close_plot_window(self, plot_window):
        if plot_window in self.plot_windows:
            self.plot_windows.remove(plot_window)
        plot_window.window.destroy()

class PlotWindow:
    def __init__(self, app, plot_type):
        self.app = app
        self.plot_type = plot_type
        self.window = tk.Toplevel(app.root)
        self.window.title(plot_type)

        # Create Figure and Canvas
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        # Add a toolbar frame
        toolbar_frame = tk.Frame(self.window)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        # Add Trace Button
        add_trace_button = tk.Button(toolbar_frame, text="Add Trace", command=self.add_trace)
        add_trace_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Traces list
        self.traces = []

        # Plot the initial data
        self.update_plot()

        # Event connections
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('pick_event', self.on_pick_event)
        self.canvas.mpl_connect('scroll_event', self.on_scroll_event)

        # Variables for moving traces
        self.active_trace = None
        self.press_event = None

    def update_plot(self):
        self.ax.cla()
        if self.plot_type == "Analog Signal":
            self.plot_analog_signal()
        elif self.plot_type == "Sampled and Quantized Signal":
            self.plot_sampled_quantized_signal()
        elif self.plot_type == "FFT Analysis":
            self.plot_fft()
        elif self.plot_type == "I and Q Components":
            self.plot_iq_components()
        elif self.plot_type == "Amplitude and Phase":
            self.plot_amplitude_phase()

        # Re-add traces after plot update
        for trace in self.traces:
            trace_line = self.ax.axvline(trace['x'], color='red', linestyle='--')
            trace['line'] = trace_line
            # Add vertical label
            label_text = trace['label_text']
            trace_label = self.ax.text(trace['x'], trace['y'], label_text, color='blue', fontsize=9,
                                       ha='left', va='bottom', rotation=90, rotation_mode='anchor', picker=True)
            trace['label'] = trace_label

        self.canvas.draw()

    def plot_analog_signal(self):
        t = np.arange(0, self.app.duration, 1 / (10 * self.app.fs))
        analog_signal = np.cos(2 * np.pi * self.app.f_signal * t) * np.cos(2 * np.pi * self.app.f_carrier * t)
        self.ax.plot(t * 1e6, analog_signal)
        self.ax.set_title('Analog Signal (Time Domain)')
        self.ax.set_xlabel('Time (µs)')
        self.ax.set_ylabel('Amplitude')
        self.ax.grid(True)
        self.data_x = t * 1e6  # Store x data for snapping
        self.data_y = analog_signal  # Store y data for labels

    def plot_sampled_quantized_signal(self):
        t_sampled, quantized_signal = self.sample_and_quantize()
        self.ax.stem(t_sampled * 1e6, quantized_signal, basefmt=" ")
        self.ax.set_title(f'Sampled and Quantized Signal at {self.app.fs/1e6} MSPS with {self.app.data_width}-bit Data Width')
        self.ax.set_xlabel('Time (µs)')
        self.ax.set_ylabel('Amplitude')
        self.ax.grid(True)
        self.data_x = t_sampled * 1e6  # Store x data for snapping
        self.data_y = quantized_signal  # Store y data for labels

    def plot_fft(self):
        _, quantized_signal = self.sample_and_quantize()
        fft_signal = np.fft.fft(quantized_signal, n=self.app.N_fft)
        fft_magnitude = np.abs(fft_signal) / self.app.N_fft
        freq_axis = np.linspace(0, self.app.fs, self.app.N_fft)
        self.ax.plot(freq_axis / 1e6, fft_magnitude)
        self.ax.set_title(f'FFT of Quantized Signal at {self.app.fs/1e6} MSPS')
        self.ax.set_xlabel('Frequency (MHz)')
        self.ax.set_ylabel('Magnitude')
        self.ax.grid(True)
        self.ax.set_xlim(0, self.app.fs / 2e6)  # Nyquist frequency
        self.data_x = freq_axis / 1e6  # Store x data for snapping
        self.data_y = fft_magnitude  # Store y data for labels

    def plot_iq_components(self):
        t_sampled, quantized_signal = self.sample_and_quantize()
        I_filtered, Q_filtered = self.extract_iq_components(quantized_signal, t_sampled)
        self.ax.plot(t_sampled * 1e6, I_filtered, label='I Component')
        self.ax.plot(t_sampled * 1e6, Q_filtered, label='Q Component')
        self.ax.set_title(f'I and Q Components at {self.app.fs/1e6} MSPS')
        self.ax.set_xlabel('Time (µs)')
        self.ax.set_ylabel('Amplitude')
        self.ax.legend()
        self.ax.grid(True)
        self.data_x = t_sampled * 1e6  # Store x data for snapping
        self.data_y = I_filtered  # Store y data for labels (using I component)

    def plot_amplitude_phase(self):
        t_sampled, quantized_signal = self.sample_and_quantize()
        I_filtered, Q_filtered = self.extract_iq_components(quantized_signal, t_sampled)
        amplitude = np.sqrt(I_filtered**2 + Q_filtered**2)
        phase = np.arctan2(Q_filtered, I_filtered)
        phase_unwrapped = np.unwrap(phase)

        # Plotting Amplitude and Phase
        self.ax.plot(t_sampled * 1e6, amplitude, label='Amplitude')
        self.ax.set_title('Amplitude and Phase of the Signal')
        self.ax.set_xlabel('Time (µs)')
        self.ax.set_ylabel('Amplitude')
        self.ax.grid(True)
        self.ax2 = self.ax.twinx()
        self.ax2.plot(t_sampled * 1e6, phase_unwrapped, color='orange', label='Phase')
        self.ax2.set_ylabel('Phase (Radians)')
        self.ax2.legend(loc='upper right')
        self.ax.legend(loc='upper left')
        self.data_x = t_sampled * 1e6  # Store x data for snapping
        self.data_y = amplitude  # Store y data for labels

    def sample_and_quantize(self):
        Ts = 1 / self.app.fs
        n_samples = int(self.app.duration / Ts)
        n = np.arange(n_samples)
        t_sampled = n * Ts
        sampled_signal = np.cos(2 * np.pi * self.app.f_signal * t_sampled) * np.cos(2 * np.pi * self.app.f_carrier * t_sampled)
        quantized_signal = self.quantize(sampled_signal, self.app.data_width)
        return t_sampled, quantized_signal

    def quantize(self, signal, bits):
        max_val = 2 ** (bits - 1) - 1
        min_val = -2 ** (bits - 1)
        signal_norm = signal / np.max(np.abs(signal))
        quantized_signal = np.round(signal_norm * max_val)
        quantized_signal = np.clip(quantized_signal, min_val, max_val)
        return quantized_signal / max_val

    def extract_iq_components(self, quantized_signal, t_sampled):
        # Adjusted the calculation to prevent Q from being zero
        I = quantized_signal * np.cos(2 * np.pi * self.app.f_carrier * t_sampled)
        Q = quantized_signal * np.sin(2 * np.pi * self.app.f_carrier * t_sampled)
        I_filtered = self.low_pass_filter(I)
        Q_filtered = self.low_pass_filter(Q)
        return I_filtered, Q_filtered

    def low_pass_filter(self, signal):
        # Implement a Butterworth low-pass filter
        if self.app.lowcut > 0:
            b, a = butter(self.app.butter, [self.app.lowcut, self.app.highcut], btype='bandpass', analog=False, fs=self.app.fs)
        else:
            b, a = butter(self.app.butter, self.app.highcut, btype='lowpass', analog=False, fs=self.app.fs)
        filtered_signal = lfilter(b, a, signal)
        return filtered_signal

    def add_trace(self):
        # Add a trace at the center of the current x-limits
        x_center = (self.ax.get_xlim()[0] + self.ax.get_xlim()[1]) / 2
        # Snap to nearest data point
        if hasattr(self, 'data_x'):
            idx = np.abs(self.data_x - x_center).argmin()
            snapped_x = self.data_x[idx]
            snapped_y = self.data_y[idx]
        else:
            snapped_x = x_center
            snapped_y = 0
        trace_line = self.ax.axvline(snapped_x, color='red', linestyle='--')
        # Add vertical label
        label_text = f"({snapped_x:.2f}, {snapped_y:.2f})"
        trace_label = self.ax.text(snapped_x, snapped_y, label_text, color='blue', fontsize=9,
                                   ha='left', va='bottom', rotation=90, rotation_mode='anchor', picker=True)
        trace = {'x': snapped_x, 'line': trace_line, 'label': trace_label, 'label_text': label_text, 'y': snapped_y}
        self.traces.append(trace)
        self.canvas.draw()

    def on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left-click
            # Check if click is near any trace
            for trace in self.traces:
                x = trace['x']
                if abs(event.xdata - x) < (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.01:
                    self.active_trace = trace
                    self.press_event = event
                    break

    def on_mouse_release(self, event):
        if self.active_trace is not None:
            self.active_trace = None
            self.press_event = None

    def on_mouse_move(self, event):
        if self.active_trace is not None and event.inaxes == self.ax:
            dx = event.xdata - self.press_event.xdata
            new_x = self.active_trace['x'] + dx
            # Snap to nearest data point
            if hasattr(self, 'data_x'):
                idx = np.abs(self.data_x - new_x).argmin()
                snapped_x = self.data_x[idx]
                snapped_y = self.data_y[idx]
                self.active_trace['x'] = snapped_x
                self.active_trace['y'] = snapped_y
                self.active_trace['line'].set_xdata([snapped_x, snapped_x])
                # Update label
                label_text = f"({snapped_x:.2f}, {snapped_y:.2f})"
                self.active_trace['label_text'] = label_text
                self.active_trace['label'].set_text(label_text)
                self.active_trace['label'].set_position((snapped_x, snapped_y))
            else:
                self.active_trace['x'] = new_x
                self.active_trace['line'].set_xdata([new_x, new_x])
                # Update label
                label_text = f"({new_x:.2f}, {0:.2f})"
                self.active_trace['label_text'] = label_text
                self.active_trace['label'].set_text(label_text)
                self.active_trace['label'].set_position((new_x, 0))
            self.canvas.draw()
            self.press_event = event

    def on_pick_event(self, event):
        # Check if the picked object is a label
        for trace in self.traces:
            if event.artist == trace['label']:
                self.edit_trace_x_value(trace)
                break

    def edit_trace_x_value(self, trace):
        # Open a dialog to input new x value
        input_window = tk.Toplevel(self.window)
        input_window.title("Edit X Value")

        tk.Label(input_window, text="Enter new X value:").grid(row=0, column=0, padx=5, pady=5)
        x_var = tk.DoubleVar(value=trace['x'])
        x_entry = tk.Entry(input_window, textvariable=x_var)
        x_entry.grid(row=0, column=1, padx=5, pady=5)

        def apply_new_x():
            try:
                new_x = x_var.get()
                # Snap to nearest data point
                if hasattr(self, 'data_x'):
                    idx = np.abs(self.data_x - new_x).argmin()
                    snapped_x = self.data_x[idx]
                    snapped_y = self.data_y[idx]
                    trace['x'] = snapped_x
                    trace['y'] = snapped_y
                    trace['line'].set_xdata([snapped_x, snapped_x])
                    # Update label
                    label_text = f"({snapped_x:.2f}, {snapped_y:.2f})"
                    trace['label_text'] = label_text
                    trace['label'].set_text(label_text)
                    trace['label'].set_position((snapped_x, snapped_y))
                else:
                    trace['x'] = new_x
                    trace['line'].set_xdata([new_x, new_x])
                    # Update label
                    label_text = f"({new_x:.2f}, {0:.2f})"
                    trace['label_text'] = label_text
                    trace['label'].set_text(label_text)
                    trace['label'].set_position((new_x, 0))
                self.canvas.draw()
                input_window.destroy()
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid numeric value.")

        apply_button = tk.Button(input_window, text="Apply", command=apply_new_x)
        apply_button.grid(row=1, column=0, columnspan=2, pady=10)

    def on_scroll_event(self, event):
        # Zoom factor
        base_scale = 1.1
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location

        if event.inaxes == self.ax:
            # If scrolling over the plot area, zoom both axes equally
            if event.button == 'up':
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                # No need to handle other cases
                return

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
            self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
            self.canvas.draw()
        elif event.inaxes is None:
            # Check if scrolling over the x-axis or y-axis areas
            # We can estimate based on the mouse position
            if event.button == 'up':
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                return

            x, y = event.x, event.y
            bbox = self.ax.get_window_extent()
            if bbox.y0 <= y <= bbox.y1:
                # Scrolling over y-axis region (left or right of the plot)
                # Zoom y-axis
                ydata = (cur_ylim[0] + cur_ylim[1]) / 2
                new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
                self.ax.set_ylim([ydata - new_height / 2, ydata + new_height / 2])
                self.canvas.draw()
            elif bbox.x0 <= x <= bbox.x1:
                # Scrolling over x-axis region (below or above the plot)
                # Zoom x-axis
                xdata = (cur_xlim[0] + cur_xlim[1]) / 2
                new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                self.ax.set_xlim([xdata - new_width / 2, xdata + new_width / 2])
                self.canvas.draw()

# Main Execution
if __name__ == "__main__":
    root = tk.Tk()
    app = RFSignalApp(root)
    root.mainloop()
