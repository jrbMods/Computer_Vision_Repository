import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox


class MotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Detection App")

        self.cap = None
        self.running = False

        # ================= UI BUTTONS =================
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)

        self.btn_webcam = tk.Button(
            self.btn_frame, text="Use Webcam", width=20, command=self.use_webcam
        )
        self.btn_webcam.grid(row=0, column=0, padx=5)

        self.btn_open_video = tk.Button(
            self.btn_frame, text="Open Video File", width=20, command=self.open_video
        )
        self.btn_open_video.grid(row=0, column=1, padx=5)

        self.btn_change_video = tk.Button(
            self.btn_frame, text="Open Different Video", width=20, command=self.open_video
        )
        self.btn_change_video.grid(row=0, column=2, padx=5)
        self.btn_change_video.grid_remove()

        self.btn_quit = tk.Button(
            self.btn_frame, text="Quit", width=20, command=self.quit_app
        )
        self.btn_quit.grid(row=0, column=3, padx=5)

        # ================= FOOTER =================
        self.footer = tk.Frame(root)
        self.footer.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.footer_label = tk.Label(
            self.footer,
            text="Created by Grigorios K. Makris / jrbMods Development / 2025-26",
            anchor="e",
            fg="gray"
        )
        self.footer_label.pack(side=tk.RIGHT)

        # ============== BACKGROUND SUBTRACTOR ==========
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )

    # ================= SAFE STOP =================
    def stop_capture(self):
        self.running = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()

    # ================= MODES =================
    def use_webcam(self):
        self.stop_capture()
        self.btn_change_video.grid_remove()
        self.start_capture(0)

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )
        if path:
            self.stop_capture()
            self.btn_change_video.grid()
            self.start_capture(path)

    # ================= CAPTURE =================
    def start_capture(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video source")
            self.cap = None
            return

        self.running = True
        self.process_frames()

    # ================= PROCESS =================
    def process_frames(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_capture()
            return

        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Motion Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # ============ RESIZE & COMBINE ============
        scale = 0.6
        frame_small = cv2.resize(frame, None, fx=scale, fy=scale)
        fg_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        fg_small = cv2.resize(fg_colored, None, fx=scale, fy=scale)

        combined = np.hstack((frame_small, fg_small))

        cv2.imshow("Original | Motion Mask", combined)

        # ESC closes only the video/camera, not the app
        if cv2.waitKey(1) & 0xFF == 27:
            self.stop_capture()
            return

        self.root.after(10, self.process_frames)

    # ================= QUIT =================
    def quit_app(self):
        self.stop_capture()
        self.root.destroy()


# ================= MAIN =================
if __name__ == "__main__":
    root = tk.Tk()
    app = MotionDetectionApp(root)
    root.mainloop()
