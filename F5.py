import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import glob, os, json, unicodedata
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
import requests
from io import BytesIO
import threading

def start_webcam():
    while True:
        # Nhận diện món ăn
        capture_from_webcam()

thread = threading.Thread(target=start_webcam)
thread.daemon = True
thread.start()  # Chạy trên luồng riêng để tránh lag giao diện chính
# === Load mô hình YOLO và CNN ===
model = YOLO('yolov10n.pt')
cnn_model = load_model('cnn_model1.keras')

classes = ['canh cải', 'canh chua', 'cá hú kho', 'cơm', 'gà chiên', 'rau muống xào',
           'thịt kho', 'thịt kho trứng', 'trứng chiên', 'đậu hủ sốt cà chua']

def normalize_text(text):
    return unicodedata.normalize('NFC', text)

with open('menu.json', 'r', encoding='utf-8') as f:
    menu_raw = json.load(f)
menu = {normalize_text(k): v for k, v in menu_raw.items()}

# === Khai báo biến toàn cục ===
root = tk.Tk()
dish_data = []
total = 0
cart_items = []
cart_total = tk.DoubleVar(value=0.0)
cart_subtotal = tk.DoubleVar(value=0.0)
cart_tax = tk.DoubleVar(value=0.0)

# --- Biến toàn cục ---
account_visible = False
customer_name = tk.StringVar(value="")   # Tên khách hàng


# --- Hàm toggle account ---
def toggle_account_info():
    global account_visible
    if account_visible:
        account_frame.place_forget()
    else:
        account_frame.place(x=950, y=60)
        account_frame.lift()
    account_visible = not account_visible




# --- Header ---
header = tk.Frame(root, bg="lightblue", height=60)
header.pack(fill=tk.X, side=tk.TOP)

logo_label = tk.Label(header, text="🍽 Food5 ", bg="orange", font=("Segoe UI", 16, "bold"))
logo_label.pack(side=tk.LEFT, padx=20)

account_btn = tk.Button(header, text="👤 My Account", command=toggle_account_info)
account_btn.pack(side=tk.RIGHT, padx=20)

# --- Account Frame ---
account_frame = tk.Frame(root, bg="white", bd=2, relief=tk.RAISED)
account_frame.configure(width=250, height=100)


account_frame.pack_propagate(False)  # Giữ kích thước cố định


# === Hàm xử lý ảnh đầu vào ===
def process_image(image_path):
    global dish_data, total
    display_selected_image(image_path)

    # Dò tìm món ăn trong ảnh sử dụng YOLO
    model(source=image_path, save=True, save_crop=True, imgsz=640, conf=0.15)

    # Lấy folder crop ảnh mới nhất
    cropped_dirs = sorted(glob.glob('runs/detect/predict*/crops'), key=os.path.getmtime, reverse=True)
    if not cropped_dirs:
        print("Không tìm thấy ảnh crop.")
        return
    latest_cropped = cropped_dirs[0]

    dish_data.clear()
    total = 0

    for folder in os.listdir(latest_cropped):
        folder_path = os.path.join(latest_cropped, folder)
        if not os.path.isdir(folder_path):
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            # Tiền xử lý ảnh cho CNN
            img = image.load_img(img_path, target_size=(200, 200))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = cnn_model.predict(img_array)
            class_idx = np.argmax(prediction)
            dish_name = classes[class_idx]

            price = menu.get(normalize_text(dish_name), 0)

            total += price
            dish_data.append((img_path, dish_name, price))

    update_ui()

# === Hiển thị ảnh đã chọn lên giao diện ===
def display_selected_image(image_path):
    pil_img = Image.open(image_path).resize((320, 240))
    tk_img = ImageTk.PhotoImage(pil_img)
    image_preview_label.configure(image=tk_img)
    image_preview_label.image = tk_img

# === Mở file ảnh từ máy ===
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        process_image(file_path)

# === Chụp ảnh từ webcam và xử lý ===
def capture_from_webcam():
    global webcam_active
    ret, frame = cap.read()
    if ret:
        webcam_active = False  # 🚫 Tạm dừng webcam

        image_path = 'captured.jpg'
        cv2.imwrite(image_path, frame)

        # Hiển thị ảnh vừa chụp
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((540, 380))
        imgtk = ImageTk.PhotoImage(image=img)
        image_preview_label.imgtk = imgtk
        image_preview_label.configure(image=imgtk)

        # Tiếp tục xử lý nhận diện
        process_image(image_path)

# === Luồng cập nhật webcam liên tục ===
webcam_active = True
def update_webcam():
    if not webcam_active:
        return
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((540, 380))
        imgtk = ImageTk.PhotoImage(image=img)
        image_preview_label.imgtk = imgtk
        image_preview_label.configure(image=imgtk)
    root.after(10, update_webcam)


# === Cập nhật UI món ăn đã nhận diện ===
def update():
    # Cập nhật dữ liệu
    root.after(50, update)  # Cập nhật mỗi 50ms, tránh lag
def update_ui():
    global total
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    cart_items.clear()
    total = 0

    for img_path, dish_name, price in dish_data:
        item_frame = ttk.Frame(scrollable_frame, padding=10)
        item_frame.pack(fill=tk.X)

        cart_items.append([dish_name, price, 1, img_path])

        pil_img = Image.open(img_path).resize((100, 100))
        tk_img = ImageTk.PhotoImage(pil_img)

        img_label = ttk.Label(item_frame, image=tk_img)
        img_label.image = tk_img
        img_label.pack(side=tk.LEFT)

        name_label = ttk.Label(item_frame, text=f"{dish_name} - {price:,} VND", font=("Segoe UI", 14))
        name_label.pack(side=tk.LEFT, padx=10)

        total += price

    
    update_cart_display()

# === Cập nhật hiển thị giỏ hàng ===
def update_cart_display():
    cart_listbox.delete(0, tk.END)
    subtotal = sum([item[1] * item[2] for item in cart_items])
    tax = round(subtotal * 0.08, 2)
    total_price = subtotal + tax

    for name, price, qty, _ in cart_items:
        cart_listbox.insert(tk.END, f"{name} (x{qty}) - {price * qty:,.0f} VND")

    cart_subtotal.set(f"{subtotal:,.0f} VND")
    cart_tax.set(f"{tax:,.0f} VND")
    cart_total.set(f"{total_price:,.0f} VND")

# === Các nút thao tác giỏ hàng ===
def remove_selected():
    selected = cart_listbox.curselection()
    if selected:
        cart_items.pop(selected[0])
        update_cart_display()

def update_quantity():
    selected = cart_listbox.curselection()
    if selected:
        cart_items[selected[0]][2] += 1
        update_cart_display()



# === Hiển thị cửa sổ QR code lấy từ URL ===
def show_qr_code_from_file(file_path):
    if not os.path.exists(file_path):
        messagebox.showerror("Lỗi", "Không tìm thấy mã QR.")
        return

    qr_window = tk.Toplevel(root)
    qr_window.title("Mã QR Thanh Toán")
    qr_window.geometry("300x300")
    qr_window.resizable(False, False)

    img = Image.open(file_path)
    img = img.resize((280, 280))
    tk_img = ImageTk.PhotoImage(img)

    label = tk.Label(qr_window, image=tk_img)
    label.image = tk_img
    label.pack(padx=10, pady=10)

      

        
# === Cập nhật UI món ăn đã nhận diện ===
def update_ui():
    global total
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    cart_items.clear()
    total = 0

    for img_path, dish_name, price in dish_data:
        item_frame = ttk.Frame(scrollable_frame, padding=10)
        item_frame.pack(fill=tk.X)

        cart_items.append([dish_name, price, 1, img_path])

        pil_img = Image.open(img_path).resize((100, 100))
        tk_img = ImageTk.PhotoImage(pil_img)

        img_label = ttk.Label(item_frame, image=tk_img)
        img_label.image = tk_img
        img_label.pack(side=tk.LEFT)

        name_label = ttk.Label(item_frame, text=f"{dish_name} - {price:,} VND", font=("Segoe UI", 14))
        name_label.pack(side=tk.LEFT, padx=10)

        total += price

    update_cart_display()

# === Cập nhật hiển thị giỏ hàng ===
def update_cart_display():
    cart_listbox.delete(0, tk.END)
    subtotal = sum([item[1] * item[2] for item in cart_items])
    tax = round(subtotal * 0.08, 2)
    total_price = subtotal + tax

    for name, price, qty, _ in cart_items:
        cart_listbox.insert(tk.END, f"{name} (x{qty}) - {price * qty:,.0f} VND")

    cart_subtotal.set(f"{subtotal:,.0f} VND")
    cart_tax.set(f"{tax:,.0f} VND")
    cart_total.set(f"{total_price:,.0f} VND")

# === Các nút thao tác giỏ hàng ===
def remove_selected():
    selected = cart_listbox.curselection()
    if selected:
        cart_items.pop(selected[0])
        update_cart_display()
        
def update_quantity():
    selected = cart_listbox.curselection()
    if selected:
        cart_items[selected[0]][2] += 1
        update_cart_display()

def cancel_order():
    global webcam_active
    if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn hủy đơn hàng không?"):
        cart_items.clear()
        update_cart_display()
        
    
        
        

def checkout():
    global webcam_active
    subtotal = sum([item[1] * item[2] for item in cart_items])
    tax = round(subtotal * 0.08, 2)
    total_price = subtotal + tax
    messagebox.showinfo("Thanh toán", f"Tổng tiền phải thanh toán: {total_price:,.0f} VND\nCảm ơn bạn đã sử dụng dịch vụ!")

    qr_path = "qr_bank.jpg"
    show_qr_code_from_file(qr_path)

    cart_items.clear()
    update_cart_display()

    
    

def logout():
    messagebox.showinfo("Logout", "Bạn đã đăng xuất!")
    #"""Hàm xử lý đăng xuất và xóa bộ nhớ"""
    mssv_label.config(text="MSSV: ")  # Xóa nội dung đã nhập
    name_label.config(text="Họ tên: ")  # Xóa nội dung đã nhập   

    
    

def reset_detection():
    """Hàm reset danh sách món ăn nhận diện"""
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    
def reset_all():
    global webcam_active

    # Xóa toàn bộ món ăn đã nhận diện
    dish_data.clear()
    cart_items.clear()
    update_cart_display()

    # Xóa nội dung hiển thị trong vùng món ăn
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    # Xóa thư mục crop YOLO cũ (nếu muốn)
    for folder in glob.glob('runs/detect/predict*'):
        try:
            import shutil
            shutil.rmtree(folder)
        except Exception as e:
            print(f"Lỗi khi xóa folder {folder}: {e}")

    # Kích hoạt lại webcam
    webcam_active = True
    update_webcam()


# === Giao diện chính ===
root.title("🍽 Automatic calculate food bill")
root.geometry("1200x800")
root.configure(bg="#000080")

# Biến lưu thông tin tổng tiền
cart_items = []
cart_total = tk.DoubleVar(value=0.0)
cart_subtotal = tk.DoubleVar(value=0.0)
cart_tax = tk.DoubleVar(value=0.0)

style = ttk.Style()
style.configure("TFrame", background="#f0f0f0")
style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 11))
style.configure("TButton", font=("Segoe UI", 11), padding=6)
style.configure("TLabelframe", font=("Segoe UI", 12, "bold"))
style.configure("TLabelframe.Label", background="#f0f0f0", foreground="#333333")

# === Webcam init ===
cap = cv2.VideoCapture(0)

# === VÙNG CHÍNH ===
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# === VÙNG TRÁI ===
left_frame = ttk.Frame(main_frame)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)


# Khung ngang chứa webcam và món ăn nhận diện
top_horizontal_frame = ttk.Frame(left_frame)
top_horizontal_frame.pack(fill=tk.BOTH, expand=True)

# Webcam view (thu nhỏ)
preview_frame = ttk.LabelFrame(top_horizontal_frame, text="📷 Webcam (320x240)")
preview_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

image_preview_label = ttk.Label(preview_frame)
image_preview_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

# Món ăn nhận diện (chuyển sang bên trái cạnh webcam)
detection_frame = ttk.LabelFrame(top_horizontal_frame, text="🍛 Món ăn nhận diện")
detection_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

canvas = tk.Canvas(detection_frame, background="#ffffff", highlightthickness=0)
scrollbar = ttk.Scrollbar(detection_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)


# Khung combo menu dưới webcam
combo_frame = ttk.LabelFrame(left_frame, text="🍱 Combo Menu")
combo_frame.pack(padx=5, pady=10, fill=tk.X)

# Danh sách combo mẫu
combos = [
    {"name": "Combo A", "items": ["Cơm trắng", "Rau muống xào", "Thịt kho trứng"], "price": "35.000 VND", "img": "image 20.png"},
    {"name": "Combo B", "items": ["Cơm trắng", "Rau muống xào", "Đậu hủ sốt cà"], "price": "30.000 VND", "img": "image 21.png"},
    {"name": "Combo C", "items": ["Cơm trắng", "Rau muống xào", "Gà chiên"], "price": "38.000 VND", "img": "mẫu.jpg"},
    {"name": "Combo D", "items": ["Cơm trắng", "Rau muống xào", "Trứng chiên"], "price": "38.000 VND", "img": "14.jpg"},
]
def create_combo_card(parent, combo):
    card = ttk.Frame(parent, relief="ridge", borderwidth=2, padding=5)
    card.pack(side=tk.LEFT, padx=10, pady=5)

    # Ảnh combo
    if os.path.exists(combo["img"]):
        img = Image.open(combo["img"]).resize((80, 80))
        photo = ImageTk.PhotoImage(img)
        label_img = ttk.Label(card, image=photo)
        label_img.image = photo
        label_img.pack()

    # Tên combo
    ttk.Label(card, text=combo["name"], font=("Segoe UI", 12, "bold")).pack(pady=2)

    # Món ăn combo (gói gọn)
    ttk.Label(card, text=", ".join(combo["items"]), font=("Segoe UI", 9), wraplength=150).pack()

    # Giá combo
    #ttk.Label(card, text=combo["price"], foreground="green", font=("Segoe UI", 10, "bold")).pack(pady=2)

for combo in combos:
    create_combo_card(combo_frame, combo)


# Nút chức năng
btn_frame = ttk.Frame(left_frame)
btn_frame.pack(pady=10)

ttk.Button(btn_frame, text="📸 Let's Check", command=capture_from_webcam).grid(row=0, column=0, padx=12)
ttk.Button(btn_frame, text="📁 Upload Image", command=upload_image).grid(row=0, column=1, padx=12)
ttk.Button(btn_frame, text="Reset", command=reset_all).grid(row=0, column=2, padx=12)

# === VÙNG PHẢI ===
right_frame = ttk.Frame(main_frame, width=480)
right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20)
right_frame.pack_propagate(False)  # Giữ kích thước cố định




# --- Customer Info Frame (trên giỏ hàng, bên phải) ---
customer_info_frame = tk.Frame(right_frame, bg="white")
customer_info_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

# Entry nhập thông tin MSSV & họ tên
tk.Label(customer_info_frame, text="Enter your student ID:", bg="white").pack(anchor='w', padx=5)
mssv_entry = tk.Entry(customer_info_frame)
mssv_entry.pack(fill=tk.X, padx=5)

tk.Label(customer_info_frame, text="Enter your name:", bg="white").pack(anchor='w', padx=5, pady=(5, 0))
name_entry = tk.Entry(customer_info_frame)
name_entry.pack(fill=tk.X, padx=5)

# Hàm xử lý xác nhận
def confirm_customer_info():
    mssv = mssv_entry.get().strip()
    name = name_entry.get().strip()
    if mssv:
        mssv_label.config(text=f"MSSV: {mssv}")
    if name:
        name_label.config(text=f"Họ tên: {name}")

# Nút xác nhận
tk.Button(customer_info_frame, text="Confirm", command=confirm_customer_info).pack(fill=tk.X, padx=5, pady=5)




# Hiển thị thông tin đã nhập
mssv_label = tk.Label(account_frame, text="MSSV: ", bg="white")
mssv_label.pack(anchor='w', padx=10)

name_label = tk.Label(account_frame, text="Họ tên: ", bg="white")
name_label.pack(anchor='w', padx=10)

points_label = tk.Label(account_frame, text="⭐ Điểm thưởng: 0", bg="white")
points_label.pack(anchor='w', padx=10)

# Điểm tích lũy và nút đăng xuất

tk.Button(account_frame, text="Log-out", bg="red", fg="white", command=logout).pack(fill=tk.X, padx=10, pady=(0, 10))






# Giỏ hàng
cart_frame = ttk.LabelFrame(right_frame, text="🛒 Cart")
cart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

cart_listbox = tk.Listbox(cart_frame, height=10, font=("Segoe UI", 11))
cart_listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Các nút thao tác giỏ hàng
button_frame = tk.Frame(cart_frame)
button_frame.pack(pady=5)

ttk.Button(button_frame, text="🗑 Remove", command=remove_selected).grid(row=0, column=0, padx=10)
ttk.Button(button_frame, text="➕ Add", command=update_quantity).grid(row=0, column=1, padx=10)
ttk.Button(button_frame, text="💳 Pay", command=checkout).grid(row=1, column=1, padx=10)
ttk.Button(button_frame, text="❌ Cancel",  command=cancel_order).grid(row=1, column=0, padx=10)

# Tổng tiền chi tiết
total_info_frame = ttk.Frame(cart_frame)
total_info_frame.pack(pady=5)

ttk.Label(total_info_frame, text="Subtotal:").grid(row=0, column=0, padx=8, sticky='e')
ttk.Label(total_info_frame, textvariable=cart_subtotal).grid(row=0, column=1, sticky='w')
ttk.Label(total_info_frame, text="VAT (8%):").grid(row=1, column=0, padx=8, sticky='e')
ttk.Label(total_info_frame, textvariable=cart_tax).grid(row=1, column=1, sticky='w')
ttk.Label(total_info_frame, text="🧾 Total:", font=("Segoe UI", 12, "bold")).grid(row=2, column=0, padx=8, sticky='e')
ttk.Label(total_info_frame, textvariable=cart_total, foreground="red", font=("Segoe UI", 14, "bold")).grid(row=2, column=1, sticky='w')

# Bắt đầu webcam
update_webcam()
root.mainloop()
cap.release()
cv2.destroyAllWindows()



