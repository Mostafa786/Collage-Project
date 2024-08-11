import pandas as pd
import numpy as np
from tkinter import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
#Eng Mostafa Mahmoud
## Metrics
from sklearn import metrics

## Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE


## Models

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from tkinter import filedialog

root = Tk()
root.geometry("1000x600")
root.title("Project")
global df
global X_resampled , Y_resampled
X_resampled=np.nan
Y_resampled=np.nan
def root1():
    def log():
        global df
        df=pd.read_csv(filedialog.askopenfilename())
        print(df)
        print("-----------------------------")
        return df

    def clean():
        def imp():
            ## impute data to solve missing values
            global df
            Stra=en3.get()
            imputer = SimpleImputer(strategy=Stra)
            imputed_df = imputer.fit_transform(df)
            df = pd.DataFrame(imputed_df, columns=df.columns)
            print(df)
        cle=Tk()
        cle.geometry("1000x600")
        cle.title("Clean")
        title1 = Label(cle, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
        title1.pack(fill=X)
        f1 = Frame(cle, bg='#0B2F3A', width=1000, height=750)
        f1.place(x=0, y=35)
        en3 = Entry(f1,width=10)
        en3.place(x=100,y=133)
        Button1 = Button(f1, width=19, text="Impute Data", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=imp)
        Button1.place(x=100, y=233)
        cle.mainloop()

    def trans():
        def Enco():
            tr=en3.get()
            Encoder = LabelEncoder()
            df[tr] = Encoder.fit_transform(df[tr])
            print(df[tr])
        trans=Tk()
        trans.geometry("1000x600")
        trans.title("Transformation")
        title1 = Label(trans, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
        title1.pack(fill=X)
        f1 = Frame(trans, bg='#0B2F3A', width=1000, height=750)
        f1.place(x=0, y=35)
        en3 = Entry(f1,width=10)
        en3.place(x=100,y=133)
        Button1 = Button(f1, width=19, text="Tranform Column", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=Enco)
        Button1.place(x=100, y=233)
        trans.mainloop()

    def selection():
        ## feature selection drop 3 columns
        def do():
            global df
            se = en3.get()
            df = df.drop(se, axis=1)
            print(df)
        select = Tk()
        select.geometry("1000x600")
        select.title("Selection")
        title1 = Label(select, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
        title1.pack(fill=X)
        f1 = Frame(select, bg='#0B2F3A', width=1000, height=750)
        f1.place(x=0, y=35)
        en3 = Entry(f1, width=15)
        en3.place(x=100, y=133)
        Button1 = Button(f1, width=19, text="Remove Column", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=do)
        Button1.place(x=100, y=233)
        select.mainloop()

    def cate():
        def imp():
            global df
            ## impute data to solve missing values
            Stra=en2.get()
            imputer = SimpleImputer(strategy=Stra)
            imputed_df = imputer.fit_transform(df)
            df = pd.DataFrame(imputed_df, columns=df.columns)
            print(df)
        def Enco():
            tr=en3.get()
            Encoder = LabelEncoder()
            df[tr] = Encoder.fit_transform(df[tr])
            print(df[tr])
        cat=Tk()
        cat.geometry("1000x600")
        cat.title("Categorial Variable")
        title1 = Label(cat, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
        title1.pack(fill=X)
        f1 = Frame(cat, bg='#0B2F3A', width=1000, height=750)
        f1.place(x=0, y=35)
        en2 = Entry(f1, width=10)
        en2.place(x=500, y=133)
        Button2 = Button(f1, width=19, text="Impute Data", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=imp)
        Button2.place(x=500, y=233)
        en3 = Entry(f1,width=10)
        en3.place(x=100,y=133)
        Button1 = Button(f1, width=19, text="Tranform Column", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=Enco)
        Button1.place(x=100, y=233)
        cat.mainloop()

    def imbal():
        def spl():
            ## split data to x and y
            la=en2.get()
            x = df.drop([la], axis=1)
            y = df[la]
            return x,y
        def smot():
            global X_resampled , Y_resampled
            ## SMOTE DATA
            x,y=spl()
            smote = SMOTE()
            X_resampled, Y_resampled = smote.fit_resample(x, y)
            print(X_resampled)
            print(Y_resampled)
            return X_resampled,Y_resampled

        smo = Tk()
        smo.geometry("1000x600")
        smo.title("SMOTE")
        title1 = Label(smo, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
        title1.pack(fill=X)
        f1 = Frame(smo, bg='#0B2F3A', width=1000, height=750)
        f1.place(x=0, y=35)
        en2 = Entry(f1, width=10)
        en2.place(x=500, y=133)
        Button2 = Button(f1, width=19, text="Split Data", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=spl)
        Button2.place(x=500, y=233)
        en3 = Entry(f1, width=10)
        en3.place(x=100, y=133)
        Button1 = Button(f1, width=19, text="The Value Of Test Split", bg="#DBA901", fg="black",
                         font=("tajawal", 16, "bold"),
                         command=smot)
        Button1.place(x=100, y=233)
        smo.mainloop()
        return smot()
    def split():
        def spl():
            ## split data to x and y
            la=en2.get()
            x = df.drop([la], axis=1)
            y = df[la]
            return x,y
        def tt():
            if X_resampled!=np.nan and Y_resampled!=np.nan:
                x,y= imbal()
            else:
                x, y = spl()

            Num_Split=float(en3.get())
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=Num_Split, shuffle=True, random_state=42)
            train=x_train.shape[0]
            print(x_train.shape[0])
            test = x_test.shape[0]
            print(x_test.shape[0])
            lab0=Label(f1,text="Num of X_train")
            lab0.place(x=100,y=333)
            lab=Label(f1,text=train)
            lab.place(x=300,y=333)
            lab3=Label(f1,text="Num of X_test")
            lab3.place(x=500,y=333)
            lab2=Label(f1,text=test)
            lab2.place(x=700,y=333)
            return x_train ,x_test ,y_train ,y_test

        spli = Tk()
        spli.geometry("1000x600")
        spli.title("Split Data")
        title1 = Label(spli, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
        title1.pack(fill=X)
        f1 = Frame(spli, bg='#0B2F3A', width=1000, height=750)
        f1.place(x=0, y=35)
        en2 = Entry(f1, width=10)
        en2.place(x=500, y=133)
        Button2 = Button(f1, width=19, text="Split Data", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=spl)
        Button2.place(x=500, y=233)
        en3 = Entry(f1, width=10)
        en3.place(x=100, y=133)
        Button1 = Button(f1, width=19, text="The Value Of Test Split", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=tt)
        Button1.place(x=100, y=233)
        spli.mainloop()
        return tt
    def scale():
        # feature scaling TO set data in range between 0:1
        global df
        Scaler = StandardScaler()
        Scaler_df = Scaler.fit_transform(df)
        print(Scaler_df)
    def save():
        file = filedialog.asksaveasfilename()
        print(file)
        df.to_csv(file, encoding="utf-8", sep=",", index=False)
        print(df)

    root1 = Tk()
    root1.geometry("1000x600")
    root1.title("Preprocessing")
    title1 = Label(root1, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
    title1.pack(fill=X)
    f1 = Frame(root1, bg='#0B2F3A', width=1000, height=750)
    f1.place(x=0, y=35)
    lab1 = Label(root1, text="choose data:", font=("tajawal", 14, "bold"))
    lab1.place(x=10, y=50)

    #en1 = Entry(root1,font=5 ,width=20)
    #en1.place(x=10, y=100)

    Button0 = Button(f1, width=19, text="Enter Data", bg="orange", fg="black", font=("tajawal", 16, "bold"),command=log)
    Button0.place(x=250, y=55)

    Button1 = Button(f1, width=19, text="Data Cleaning", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),command=clean)
    Button1.place(x=100, y=133)

    Button2 = Button(f1, width=19, text="Data Transformation", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),command=trans)
    Button2.place(x=100, y=233)

    Button3 = Button(f1, width=19, text="Feature Selection", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),command=selection)
    Button3.place(x=100, y=333)

    Button4 = Button(f1, width=19, text="Categorical Variables", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),command=cate)
    Button4.place(x=100, y=433)

    Button5 = Button(f1, width=19, text="SMOTE", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),command=imbal)
    Button5.place(x=400, y=133)

    Button6 = Button(f1, width=19, text="Splitting Data", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),command=split)
    Button6.place(x=400, y=233)

    Button7 = Button(f1, width=19, text="Feature Scaling", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),command=scale )
    Button7.place(x=400, y=333)

    Button8 = Button(f1, width=19, text="Close Program", bg="red", fg="black", font=("tajawal", 16, "bold"),
                     command=quit)
    Button8.place(x=700, y=433)

    Button5 = Button(f1, width=19, text="Save", bg="green", fg="black", font=("tajawal", 16, "bold"),command=save)
    Button5.place(x=400, y=433)

    root1.mainloop()
    return split

def root2():
    def log():
        global df
        df=pd.read_csv(filedialog.askopenfilename())
        print(df)
        print("-----------------------------")
        return df
    def svm():
        def spl():
            ## split data to x and y
            la=en2.get()
            x = df.drop([la], axis=1)
            y = df[la]
            return x,y
        def tt():
            x,y=spl()
            Num_Split=float(en3.get())
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=Num_Split, shuffle=True, random_state=42)
            train=x_train.shape[0]
            print(x_train.shape[0])
            test = x_test.shape[0]
            print(x_test.shape[0])
            wr3 = Label(f1, text="x_train : ")
            wr3.place(x=500, y=250)
            la3 = Label(f1, text=train)
            la3.place(x=600, y=250)

            wr4 = Label(f1, text="x_test: ")
            wr4.place(x=500, y=300)

            la4 = Label(f1, text=test)
            la4.place(x=600, y=300)
            return x_train, x_test, y_train, y_test

        def sv():
            #x_train,x_test,y_train,y_test=root1()
            x_train, x_test, y_train, y_test = tt()
            ker=en1.get()
            clf5 = SVC(kernel=ker, C=1.0)
            clf5.fit(x_train, y_train)
            print(clf5.score(x_train,y_train))
            ji=clf5.score(x_train,y_train)
            wr1=Label(f1,text="Score : ")
            wr1.place(x=500,y=350)
            la1=Label(f1,text=ji)
            la1.place(x=600,y=350)

            y_pred5 = clf5.predict(x_test)
            #print(y_pred5)
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            accuracy = accuracy_score(y_test, y_pred5)
            precision = precision_score(y_test, y_pred5)
            recall = recall_score(y_test, y_pred5)

            print(f"accuracy: {accuracy}")
            print(f"precision: {precision}")
            print(f"recall: {recall}")
            wr2 = Label(f1, text="accuracy : ")
            wr2.place(x=500,y=400)
            la2 = Label(f1, text=accuracy)
            la2.place(x=600,y=400)

            wr3 = Label(f1, text="precision : ")
            wr3.place(x=500,y=450)
            la3 = Label(f1, text=precision)
            la3.place(x=600,y=450)

            wr4 = Label(f1, text="recall: ")
            wr4.place(x=500,y=500)

            la4 = Label(f1, text=recall)
            la4.place(x=600,y=500)
        s=Tk()
        s.geometry("1000x600")
        s.title("SVM")
        title1 = Label(s, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
        title1.pack(fill=X)
        f1 = Frame(s, bg='#0B2F3A', width=1000, height=750)
        f1.place(x=0, y=35)
        en2 = Entry(f1, width=20)
        en2.place(x=500, y=100)
        Button2 = Button(f1, width=19, text="Split Data", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=spl)
        Button2.place(x=500, y=200)
        en3 = Entry(f1, width=20)
        en3.place(x=100, y=100)
        Button1 = Button(f1, width=19, text="The Value Of Test Split", bg="#DBA901", fg="black",
                         font=("tajawal", 16, "bold"),
                         command=tt)
        Button1.place(x=100, y=200)
        en1=Entry(f1,width=20)
        en1.place(x=100,y=300)
        Button3=Button(f1, width=19, text="Kernel", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=sv)
        Button3.place(x=100,y=400)
        s.mainloop()

    def knn():
        def spl():
            ## split data to x and y
            la=en2.get()
            x = df.drop([la], axis=1)
            y = df[la]
            return x,y
        def tt():
            x,y=spl()
            Num_Split=float(en3.get())
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=Num_Split, shuffle=True, random_state=42)
            train=x_train.shape[0]
            print(x_train.shape[0])
            test = x_test.shape[0]
            print(x_test.shape[0])
            wr3 = Label(f1, text="x_train : ")
            wr3.place(x=500, y=250)
            la3 = Label(f1, text=train)
            la3.place(x=600, y=250)

            wr4 = Label(f1, text="x_test: ")
            wr4.place(x=500, y=300)

            la4 = Label(f1, text=test)
            la4.place(x=600, y=300)
            return x_train, x_test, y_train, y_test

        def kn():
            #x_train,x_test,y_train,y_test=root1()
            x_train, x_test, y_train, y_test = tt()
            num=int(en1.get())
            clf1 = KNeighborsClassifier(n_neighbors=num)
            clf1.fit(x_train, y_train)
            print(clf1.score(x_train,y_train))
            ji=clf1.score(x_train,y_train)
            wr1=Label(f1,text="Score : ")
            wr1.place(x=500,y=350)
            la1=Label(f1,text=ji)
            la1.place(x=600,y=350)

            y_pred5 = clf1.predict(x_test)
            #print(y_pred5)
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            accuracy = accuracy_score(y_test, y_pred5)
            precision = precision_score(y_test, y_pred5)
            recall = recall_score(y_test, y_pred5)

            print(f"accuracy: {accuracy}")
            print(f"precision: {precision}")
            print(f"recall: {recall}")
            wr2 = Label(f1, text="accuracy : ")
            wr2.place(x=500,y=400)
            la2 = Label(f1, text=accuracy)
            la2.place(x=600,y=400)

            wr3 = Label(f1, text="precision : ")
            wr3.place(x=500,y=450)
            la3 = Label(f1, text=precision)
            la3.place(x=600,y=450)

            wr4 = Label(f1, text="recall: ")
            wr4.place(x=500,y=500)

            la4 = Label(f1, text=recall)
            la4.place(x=600,y=500)
        k=Tk()
        k.geometry("1000x600")
        k.title("KNN")
        title1 = Label(k, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
        title1.pack(fill=X)
        f1 = Frame(k, bg='#0B2F3A', width=1000, height=750)
        f1.place(x=0, y=35)
        en2 = Entry(f1, width=20)
        en2.place(x=500, y=100)
        Button2 = Button(f1, width=19, text="Split Data", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=spl)
        Button2.place(x=500, y=200)
        en3 = Entry(f1, width=20)
        en3.place(x=100, y=100)
        Button1 = Button(f1, width=19, text="The Value Of Test Split", bg="#DBA901", fg="black",
                         font=("tajawal", 16, "bold"),
                         command=tt)
        Button1.place(x=100, y=200)
        en1=Entry(f1,width=20)
        en1.place(x=100,y=300)
        Button3=Button(f1, width=19, text="n_neighbors", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=kn)
        Button3.place(x=100,y=400)
        k.mainloop()
    def ann():
        def spl():
            ## split data to x and y
            la=en2.get()
            x = df.drop([la], axis=1)
            y = df[la]
            return x,y
        def tt():
            x,y=spl()
            Num_Split=float(en3.get())
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=Num_Split, shuffle=True, random_state=42)
            train=x_train.shape[0]
            print(x_train.shape[0])
            test = x_test.shape[0]
            print(x_test.shape[0])
            wr3 = Label(f1, text="x_train : ")
            wr3.place(x=500, y=250)
            la3 = Label(f1, text=train)
            la3.place(x=600, y=250)

            wr4 = Label(f1, text="x_test: ")
            wr4.place(x=500, y=300)

            la4 = Label(f1, text=test)
            la4.place(x=600, y=300)
            return x_train, x_test, y_train, y_test

        def an():
            #x_train,x_test,y_train,y_test=root1()
            x_train, x_test, y_train, y_test = tt()
            feature1=int(en5.get())
            feature2=en1.get()
            feature3=en4.get()

            NN = MLPClassifier(hidden_layer_sizes=feature1, activation=feature2, learning_rate=feature3)
            NN.fit(x_train, y_train)
            print(NN.score(x_train,y_train))
            ji=NN.score(x_train,y_train)
            wr1=Label(f1,text="Score : ")
            wr1.place(x=500,y=350)
            la1=Label(f1,text=ji)
            la1.place(x=600,y=350)

            y_pred5 = NN.predict(x_test)
            #print(y_pred5)
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            accuracy = accuracy_score(y_test, y_pred5)


            print(f"accuracy: {accuracy}")
            #print(f"precision: {precision}")
            #print(f"recall: {recall}")
            wr2 = Label(f1, text="accuracy : ")
            wr2.place(x=500,y=400)
            la2 = Label(f1, text=accuracy)
            la2.place(x=600,y=400)

            #wr3 = Label(f1, text="precision : ")
            #wr3.place(x=500,y=450)
            #la3 = Label(f1, text=precision)
            #la3.place(x=600,y=450)

            #wr4 = Label(f1, text="recall: ")
            #wr4.place(x=500,y=500)

            #la4 = Label(f1, text=recall)
            #la4.place(x=600,y=500)
        a=Tk()
        a.geometry("1000x600")
        a.title("ANN")
        title1 = Label(a, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
        title1.pack(fill=X)
        f1 = Frame(a, bg='#0B2F3A', width=1000, height=750)
        f1.place(x=0, y=35)
        en2 = Entry(f1, width=20)
        en2.place(x=500, y=100)
        Button2 = Button(f1, width=19, text="Split Data", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=spl)
        Button2.place(x=500, y=200)
        en3 = Entry(f1, width=20)
        en3.place(x=100, y=100)
        Button1 = Button(f1, width=19, text="The Value Of Test Split", bg="#DBA901", fg="black",
                         font=("tajawal", 16, "bold"),command=tt)
        Button1.place(x=100, y=200)

        en5 = Entry(f1, width=20)
        en5.place(x=100, y=250)
        label5 = Label(f1, text="hidden_layer_sizes")
        label5.place(x=300, y=250)

        en1=Entry(f1,width=20)
        en1.place(x=100,y=300)
        label1 = Label(f1, text="activation")
        label1.place(x=300, y=300)
        en4 = Entry(f1, width=20)
        en4.place(x=100, y=350)
        label2 = Label(f1, text="learning_rate")
        label2.place(x=300, y=350)
        Button3=Button(f1, width=19, text="Add Features", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=an)

        Button3.place(x=100,y=400)
        a.mainloop()
    def DecTree():
        def spl():
            ## split data to x and y
            la=en2.get()
            x = df.drop([la], axis=1)
            y = df[la]
            return x,y
        def tt():
            x,y=spl()
            Num_Split=float(en3.get())
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=Num_Split, shuffle=True, random_state=42)
            train=x_train.shape[0]
            print(x_train.shape[0])
            test = x_test.shape[0]
            print(x_test.shape[0])
            wr3 = Label(f1, text="x_train : ")
            wr3.place(x=500, y=250)
            la3 = Label(f1, text=train)
            la3.place(x=600, y=250)

            wr4 = Label(f1, text="x_test: ")
            wr4.place(x=500, y=300)

            la4 = Label(f1, text=test)
            la4.place(x=600, y=300)
            return x_train, x_test, y_train, y_test

        def dec():
            #x_train,x_test,y_train,y_test=root1()
            x_train, x_test, y_train, y_test = tt()
            feature1=int(en1.get())
            feature2=int(en4.get())
            clfc = DecisionTreeClassifier(random_state=0, max_depth=feature1, max_features=feature2)
            clfc.fit(x_train, y_train)
            print(clfc.score(x_train,y_train))
            ji=clfc.score(x_train,y_train)
            wr1=Label(f1,text="Score : ")
            wr1.place(x=500,y=350)
            la1=Label(f1,text=ji)
            la1.place(x=600,y=350)

            y_pred5 = clfc.predict(x_test)
            #print(y_pred5)
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            accuracy = accuracy_score(y_test, y_pred5)
            precision = precision_score(y_test, y_pred5)
            recall = recall_score(y_test, y_pred5)

            print(f"accuracy: {accuracy}")
            print(f"precision: {precision}")
            print(f"recall: {recall}")
            wr2 = Label(f1, text="accuracy : ")
            wr2.place(x=500,y=400)
            la2 = Label(f1, text=accuracy)
            la2.place(x=600,y=400)

            wr3 = Label(f1, text="precision : ")
            wr3.place(x=500,y=450)
            la3 = Label(f1, text=precision)
            la3.place(x=600,y=450)

            wr4 = Label(f1, text="recall: ")
            wr4.place(x=500,y=500)

            la4 = Label(f1, text=recall)
            la4.place(x=600,y=500)
        de=Tk()
        de.geometry("1000x600")
        de.title("Desicion Tree")
        title1 = Label(de, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
        title1.pack(fill=X)
        f1 = Frame(de, bg='#0B2F3A', width=1000, height=750)
        f1.place(x=0, y=35)
        en2 = Entry(f1, width=20)
        en2.place(x=500, y=100)
        Button2 = Button(f1, width=19, text="Split Data", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=spl)
        Button2.place(x=500, y=200)
        en3 = Entry(f1, width=20)
        en3.place(x=100, y=100)
        Button1 = Button(f1, width=19, text="The Value Of Test Split", bg="#DBA901", fg="black",
                         font=("tajawal", 16, "bold"),
                         command=tt)
        Button1.place(x=100, y=200)
        en1 = Entry(f1,width=20)
        en1.place(x=100,y=300)
        label1 = Label(f1,text="Max Depth")
        label1.place(x=300,y=300)
        en4 = Entry(f1, width=20)
        en4.place(x=100, y=350)
        label2 = Label(f1, text="Max Feature")
        label2.place(x=300, y=350)
        Button3 = Button(f1, width=19, text="Add Features", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                         command=dec)
        Button3.place(x=100,y=400)
        de.mainloop()

    root2 = Tk()
    root2.geometry("1000x600")
    root2.title("Classification")
    title1 = Label(root2, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
    title1.pack(fill=X)
    f1 = Frame(root2, bg='#0B2F3A', width=1000, height=750)
    f1.place(x=0, y=35)
    lab1 = Label(root2, text="choose data:", font=("tajawal", 14, "bold"))
    lab1.place(x=10, y=50)

    Button0 = Button(f1, width=19, text="Enter Data", bg="orange", fg="black", font=("tajawal", 16, "bold"),
                     command=log)
    Button0.place(x=100, y=55)
    Button1 = Button(f1, width=19, text="SVM", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),command=svm)
    Button1.place(x=500, y=53)

    Button2 = Button(f1, width=19, text="KNN", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),command=knn)
    Button2.place(x=500, y=153)

    Button3 = Button(f1, width=19, text="ANN", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),command=ann )
    Button3.place(x=500, y=253)

    Button4 = Button(f1, width=19, text="Decision Tree", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),command=DecTree)
    Button4.place(x=500, y=353)

    Button5 = Button(f1, width=19, text="Save", bg="green", fg="black", font=("tajawal", 16, "bold"))
    Button5.place(x=170, y=453)

    Button6 = Button(f1, width=19, text="Close Program", bg="red", fg="black", font=("tajawal", 16, "bold"),
                     command=quit)
    Button6.place(x=500, y=453)
    root2.mainloop()


def root3():
    root3 = Tk()
    root3.geometry("1000x600")
    root3.title("Clustering")
    title1 = Label(root3, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
    title1.pack(fill=X)
    f1 = Frame(root3, bg='#0B2F3A', width=1000, height=750)
    f1.place(x=0, y=35)
    Button1 = Button(f1, width=19, text="K-Means", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"))
    Button1.place(x=370, y=203)

    Button2 = Button(f1, width=19, text="Save", bg="green", fg="black", font=("tajawal", 16, "bold"))
    Button2.place(x=170, y=303)

    Button3 = Button(f1, width=19, text="Close Program", bg="red", fg="black", font=("tajawal", 16, "bold"),
                     command=quit)
    Button3.place(x=570, y=303)

    root3.mainloop()


title1 = Label(root, text="Project Assignment", fg="gold", bg="black", font=('tajawal', 18, "bold"))
title1.pack(fill=X)
f1 = Frame(root, bg='#0B2F3A', width=1000, height=750)
f1.place(x=0, y=35)

title3 = Label(f1, text="Team Member Sec(8):", bg='#0B2F3A', fg="White", font=("tajawal", 16, "bold"))
title3.place(x=750, y=20)

title4 = Label(f1, text="مصطفي محمود محمد", bg='#0B2F3A', fg="White", font=("tajawal", 16, "bold"))
title4.place(x=825, y=70)

title4 = Label(f1, text="حسام محمد محمد النجار", bg='#0B2F3A', fg="White", font=("tajawal", 16, "bold"))
title4.place(x=810, y=120)

title4 = Label(f1, text="جني اشرف عبد المنعم", bg='#0B2F3A', fg="White", font=("tajawal", 16, "bold"))
title4.place(x=820, y=170)

title4 = Label(f1, text="يحيي محمد عبد القادر", bg='#0B2F3A', fg="White", font=("tajawal", 16, "bold"))
title4.place(x=820, y=220)

title4 = Label(f1, text="اياد محمد عبد اللطيف", bg='#0B2F3A', fg="White", font=("tajawal", 16, "bold"))
title4.place(x=820, y=270)

Button4 = Button(f1, width=19, text="Preprocessing", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                 command=root1)
Button4.place(x=370, y=153)

Button5 = Button(f1, width=19, text="Classification", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                 command=root2)
Button5.place(x=370, y=253)

Button6 = Button(f1, width=19, text="Clustering", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"), command=root3)
Button6.place(x=370, y=353)

Button7 = Button(f1, width=19, text="Close Program", bg="#DBA901", fg="black", font=("tajawal", 16, "bold"),
                 command=quit)
Button7.place(x=370, y=453)
root.mainloop()
