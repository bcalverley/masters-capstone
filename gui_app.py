import tkinter as tk
from tkinter import ttk


class CardScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pokémon Card Scanner")
        self.root.geometry("500x400")
        self.root.resizable(False, False)

        # =========================
        # Main container
        # =========================
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill="both", expand=True)

        # =========================
        # Title
        # =========================
        title_label = ttk.Label(
            main_frame,
            text="Pokémon Card Scanner",
            font=("Segoe UI", 16, "bold")
        )
        title_label.pack(pady=(0, 20))

        # =========================
        # Button frame
        # =========================
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        self.capture_button = ttk.Button(
            button_frame,
            text="Capture Image",
            command=self.on_capture_clicked
        )
        self.capture_button.grid(row=0, column=0, padx=10)

        self.upload_button = ttk.Button(
            button_frame,
            text="Upload Image",
            command=self.on_upload_clicked
        )
        self.upload_button.grid(row=0, column=1, padx=10)

        # =========================
        # Result section
        # =========================
        result_frame = ttk.LabelFrame(
            main_frame,
            text="Result",
            padding=15
        )
        result_frame.pack(fill="both", expand=True, pady=20)

        self.result_label = ttk.Label(
            result_frame,
            text="Waiting for input...",
            font=("Segoe UI", 11)
        )
        self.result_label.pack()

    # =========================
    # Button callbacks
    # =========================
    def on_capture_clicked(self):
        self.result_label.config(text="Capture button clicked.\n(Camera not wired yet.)")

    def on_upload_clicked(self):
        self.result_label.config(text="Upload button clicked.\n(File picker not wired yet.)")


# =========================
# App entry point
# =========================
def main():
    root = tk.Tk()
    app = CardScannerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
