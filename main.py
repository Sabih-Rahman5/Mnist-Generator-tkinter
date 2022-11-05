import torch
from tkinter import *
from torchvision.utils import save_image


import os
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


def denorm(x):
    out = (x+1)/2
    return out.clamp(0, 1)


Generator = torch.load('models/Generator.pth', map_location='cpu')


window = Tk()
window.title("Mnist Gan GUI")
window.geometry('1000x600')


def generate_images():
    num = int(e1.get())
    idx = int(e2.get())
    row = int(e3.get())

    save_fake_images(num, idx, row)
    display_image(idx)


def save_fake_images(num, index, rows):
    sample_vectors = torch.randn(num, 64)
    fake_images = Generator(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_name = 'fake_images-{0:0=4d}.png'.format(index)
    print('saving:', fake_name)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_name), nrow=rows)


def display_image(idx):
    global img
    img = PhotoImage(file=r'samples\fake_images-{0:0=4d}.png'.format(idx))
    blank_label.config(image=img)


h = Label(window, text='MNIST-Handwritten Digits Generator', bg='blue', fg='white', font=('calibre', 20, 'bold'))
h.pack(side=TOP, fill=X)

controls = Frame(window, bg='royalblue1')
controls.pack(side=BOTTOM, fill=X)


l1 = Label(controls, text='Enter No Of Images:', bg='royalblue1', height=2, font=('calibre', 9, 'bold'))
l1.grid(row=0, column=0)
e1 = Entry(controls, bd=0, width=10)
e1.grid(row=0, column=1, padx=4)
e1.insert(END, '10')

l2 = Label(controls, text='Enter Index:', bg='royalblue1', height=2, font=('calibre', 9, 'bold'))
l2.grid(row=0, column=2, pady=5, padx=5)
e2 = Entry(controls, bd=0, width=10)
e2.grid(row=0, column=3, padx=5, pady=5)
e2.insert(END, '0')


l3 = Label(controls, text='Enter No. Of Columns:', bg='royalblue1', height=2, font=('calibre', 9, 'bold'))
l3.grid(row=0, column=4, pady=5, padx=5)
e3 = Entry(controls, bd=0, width=10)
e3.grid(row=0, column=5, padx=5, pady=5)
e3.insert(END, '10')


f2 = Frame(window, bg='green')
f2.place(x=0, y=38, width=1366, height=568)

img = PhotoImage(file=r'samples\Canvas.png').subsample(2, 3)

blank_label = Label(f2, image=img)
blank_label.place(relx=0.5, rely=0.5, anchor="center")


b1 = Button(controls, text='Generate', bd=0, bg='green', fg='white', font=('calibre', 9, 'bold'), width=20, height=5,
            command=generate_images)
b1.grid(row=0, column=7, pady=5, padx=100)

b2 = Button(controls, text='Exit', bd=0, bg='red', fg='white', command=window.destroy, width=20, height=5,
            font=('calibre', 9, 'bold'))
b2.grid(row=0, column=10, pady=5)
window.mainloop()
