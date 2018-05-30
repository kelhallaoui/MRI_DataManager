import h5py
import numpy as np
import tkinter
from tkinter import *
import tkinter as tk
from tkinter import ttk
from tkinter import Tk, Label, Button, Canvas, Radiobutton, IntVar, W, StringVar, Canvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GUI(object):

    def get_params(self):
        with h5py.File('experiments/data_tumor_large_size_variation.h5', 'r') as hf:
            print('Getting parameters...')
            attrs = {}
            for item in hf.attrs.keys():
                if 'subject' not in item: attrs.update({item: hf.attrs[item]})
            return attrs

    def get_data_info(self):
        key_dic = {}
        with h5py.File('experiments/data_tumor_large_size_variation.h5', 'r') as hf:
            keys = list(hf.keys())
            for key in keys:
                temp = key.split('_')
        
                if temp[0] not in key_dic: 
                    key_dic.update({temp[0]:{'Identifiers':[], 'Indices': 0}})
        
                if key_dic[temp[0]]['Indices'] < int(temp[-1]): 
                    key_dic[temp[0]]['Indices'] = int(temp[-1])
    
                if '_'.join(temp[1:-1]) not in key_dic[temp[0]]['Identifiers']: 
                    key_dic[temp[0]]['Identifiers'].append('_'.join(temp[1:-1]))

        return key_dic

    def sel_dataset(self):
        selection = "You selected the option " + str(self.option.get())
        print(selection)
        self.dataset = self.option.get().lower()
        self.show_data()

    def sel_identifier(self):
        selection = "You selected the option " + str(self.option.get())
        print(selection)
        self.identifier = self.option_identifier.get().lower()
        self.show_data()

    def show_data(self):
        addr = self.dataset + '_' + self.identifier + '_0'
        print(addr)

        with h5py.File('experiments/data_tumor_large_size_variation.h5', 'r') as hf:
            vals = list(hf[addr])
            data = np.asarray(vals)

        
        if self.identifier == 'image':     data = np.abs(data[0])
        elif self.identifier == 'k_space': data = np.log(np.abs(data[0]))
        else: data = np.random.randint(1, 5, size=(100,100))


        f = Figure(figsize = (3,3))
        a = f.add_subplot(111)

        a.imshow(data.T, cmap='gray', interpolation='none')

        # Labels for major ticks
        a.set_xticklabels([])
        a.set_yticklabels([])

        # Minor ticks
        a.set_xticks([])
        a.set_yticks([])

        canvas = FigureCanvasTkAgg(f, master = self.pages['Dataset'])
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=0, columnspan = 3, rowspan = 2, sticky = 'W')

    def __init__(self, master):

        height, width = 400, 400
        self.master = master
        self.master.geometry(''+str(width)+'x'+str(height))

        # Set up a notebook inside the master GUI
        nb = ttk.Notebook(root, width = width, height = height)
        nb.grid(row = 0, column=0)

        # Set up the frames within the notebook
        self.pages = {}
        for tab in ['Parameters', 'Dataset']:
            self.pages.update({tab: ttk.Frame(nb, width = width, height = height)})
            nb.add(self.pages[tab], text=tab)

        # Write all the parameters
        params = self.get_params()
        for ix, i in enumerate(params):
            param = Label(self.pages['Parameters'])
            param.config(text = i + ': ')
            param.grid(row=ix, column=0, sticky=tkinter.W) 

            val = Label(self.pages['Parameters'])
            val.config(text = params[i])
            val.grid(row=ix, column=1, sticky=tkinter.W) 


        # Set up radiobuttons for the dataset names
        header = Label(self.pages['Dataset'])
        header.config(text = 'Dataset')
        header.grid(row=0, column=0) 

        self.option = StringVar()
        self.option.set(0)

        buttons = []
        vars = []
        for idx, label in enumerate(["Train","Validation","Test"]):
            vars.append(StringVar(value=label))
            buttons.append(Radiobutton(self.pages['Dataset'], padx=20, pady=5, text=label, variable=vars[-1], 
                                       value=label, var = self.option, command = self.sel_dataset))
            buttons[-1].grid(row=0, column=idx+1) 
        buttons[0].select()


        key_dict = self.get_data_info()

        header = Label(self.pages['Dataset'])
        header.config(text = 'Identifier')
        header.grid(row=1, column=0) 


        self.option_identifier = StringVar()
        self.option_identifier.set(0)

        labels = key_dict['train']['Identifiers']
        buttons_identifier = []
        vars_identifier = []
        for idx, label in enumerate(labels):
            vars_identifier.append(StringVar(value=label))
            buttons_identifier.append(Radiobutton(self.pages['Dataset'], padx=20, pady=5, text=label, variable=vars_identifier[-1], 
                                       value=label, var = self.option_identifier, command = self.sel_identifier))
            buttons_identifier[-1].grid(row=1, column=idx+1) 
        buttons_identifier[0].select()

        b = Button(self.pages['Dataset'], text="Prev.")
        b.grid(row=2, column=idx+1, sticky = 'S')

        b = Button(self.pages['Dataset'], text="Next ")
        b.grid(row=3, column=idx+1, sticky = 'N')


        self.dataset = "train"
        self.identifier = labels[0]


        self.show_data()
        
        '''
        w = Canvas(pages['Dataset'], width=400, height=200)
        #w.create_rectangle(0, 0, 400, 200, fill="white")
        w.grid(row=2, column=0, columnspan = idx+2)
        
        fig = FigureCanvasTkAgg(f, master = w)
        fig.get_tk_widget().grid(row=0, column=0)
        fig.draw()
        '''


        #w.grid(row=2, column=0, columnspan = idx+1)



        '''
        self.label = Label(root)
        self.label.pack()

        self.button1 = Button(root, text = 'Restart')
        self.button1.config(command = self.restart)
        self.button1.place(relx = 0., rely = 0.8, relwidth=0.5, relheight=0.2)

        self.button2 = Button(root, text = 'Quit')
        self.button2.config(command = self.quit)
        self.button2.place(relx = 0.5, rely = 0.8, relwidth=0.5, relheight=0.2)

        self.master.bind("<KeyPress>", self.keydown)
        '''

    def keydown(self, e):
        print(e.keycode)
    
    def quit(self):
        self.master.destroy()
    
    def restart(self):
        pass
        
    def update_screen(self, mat):
        pass




root = Tk()
root.style = ttk.Style()
root.style.theme_use("winnative")
my_gui = GUI(root)

root.mainloop()

