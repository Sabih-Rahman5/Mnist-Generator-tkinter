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
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('saving:', fake_fname)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=rows)


def display_image(idx):
    global img
    img = PhotoImage(file=r'samples\fake_images-{0:0=4d}.png'.format(idx))
    testlabel.config(image=img)


heading = Label(window, text="MNIST Handwritten Digits Generator", bg='darkblue')
heading.pack()
f1 = Frame(window, bg='royalblue1')
f1.place(x=0, y=40, width=450, height=700)

l1 = Label(f1, text='Enter No of images:', bg='light blue')
l1.grid(row=2, column=0, pady=5, padx=3)
e1 = Entry(f1, bd=0)
e1.grid(row=2, column=1, pady=5)
e1.insert(END, '10')

l2 = Label(f1, text='Enter Index:', bg='light blue')
l2.grid(row=3, column=0, pady=5, padx=3)
e2 = Entry(f1, bd=0)
e2.grid(row=3, column=1, pady=5)
e2.insert(END, '0')


l3 = Label(f1, text='Enter no. of columns', bg='light blue')
l3.grid(row=4, column=0, pady=5, padx=3)
e3 = Entry(f1, bd=0)
e3.grid(row=4, column=1, pady=5)
e3.insert(END, '10')


f2 = Frame(window, bg='green')
f2.place(x=450, y=40, width=1000, height=700)
img = PhotoImage(file=r'samples\Canvas.png').subsample(2, 3)

testlabel = Label(f2, bg='pink', image=img)
testlabel.place(relx=0.5, rely=0.5, anchor="center")


b2 = Button(f1, text='Generate', bd=0, bg='royalblue1', fg='white', width=20, font=('calibre', 9, 'bold'),
            command=generate_images)
b2.grid(row=2, column=2, pady=5, padx=50)
window.mainloop()
